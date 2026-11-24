// Scene object materials. Wood and rock use scanned CC0 diffuse maps, sampled triplanar in world
// space (the procedural meshes have no UVs), with normal maps and parallax heights derived from
// those scans. Ambient light comes from the baked probe grid; reflections are glossy sky. The
// other classes stay procedural. Direct light is GGX; dielectric/metal F0 is energy-aware.
layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vAlbedo;
layout(location = 3) flat in float vMaterial;
layout(location = 0) out vec4 outHdr;

const float DIELECTRIC_F0 = 0.04;
const float WOOD_TILE = 1.6;   // metres per repeat of the wood scan
const float ROCK_TILE = 3.0;   // metres per repeat of the rock scan
const float WOOD_GAIN = 0.62;  // artistic scale on the scan's linear albedo (weathered deck)
const float ROCK_GAIN = 2.2;   // the rock scan is very dark (mean linear albedo about 0.04)
const float WOOD_DEPTH = 0.03; // parallax relief depth (m): full height range of the scan
const float ROCK_DEPTH = 0.15;

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
float noise3(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i + vec3(0,0,0));
    float n100 = hash31(i + vec3(1,0,0));
    float n010 = hash31(i + vec3(0,1,0));
    float n110 = hash31(i + vec3(1,1,0));
    float n001 = hash31(i + vec3(0,0,1));
    float n101 = hash31(i + vec3(1,0,1));
    float n011 = hash31(i + vec3(0,1,1));
    float n111 = hash31(i + vec3(1,1,1));
    float x00 = mix(n000, n100, u.x);
    float x10 = mix(n010, n110, u.x);
    float x01 = mix(n001, n101, u.x);
    float x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

float distributionGGX(float NoH, float alpha) {
    float a2 = alpha * alpha;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}
float smithG1(float NoX, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * NoX / max(NoX + sqrt(a2 + (1.0 - a2) * NoX * NoX), 1e-5);
}
vec3 fresnelSchlick(vec3 f0, float VoH) {
    return f0 + (1.0 - f0) * pow(1.0 - clamp(VoH, 0.0, 1.0), 5.0);
}

// Triplanar blend weights from the geometric normal: the dominant world axis wins.
vec3 triWeights(vec3 n) {
    vec3 w = pow(abs(n), vec3(4.0));
    return w / (w.x + w.y + w.z);
}

// Triplanar albedo: the three world projections, blended. Texture coordinates are world metres / tile.
vec3 triAlbedo(sampler2D t, vec3 p, vec3 w, float tileInv) {
    return texture(t, p.zy * tileInv).rgb * w.x
         + texture(t, p.xz * tileInv).rgb * w.y
         + texture(t, p.xy * tileInv).rgb * w.z;
}

// Triplanar normal map (whiteout blend). gn is the geometric normal; the result is world space.
vec3 triNormal(sampler2D t, vec3 p, vec3 gn, vec3 w, float tileInv) {
    vec3 tx = texture(t, p.zy * tileInv).xyz * 2.0 - 1.0;
    vec3 ty = texture(t, p.xz * tileInv).xyz * 2.0 - 1.0;
    vec3 tz = texture(t, p.xy * tileInv).xyz * 2.0 - 1.0;
    tx = vec3(tx.xy + gn.zy, abs(tx.z) * gn.x);
    ty = vec3(ty.xy + gn.xz, abs(ty.z) * gn.y);
    tz = vec3(tz.xy + gn.xy, abs(tz.z) * gn.z);
    return normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
}

// Height from the normal map's alpha (0..1, 0.5 = mean surface), blended like the albedo.
float triHeight(sampler2D t, vec3 p, vec3 w, float tileInv) {
    return texture(t, p.zy * tileInv).a * w.x
         + texture(t, p.xz * tileInv).a * w.y
         + texture(t, p.xy * tileInv).a * w.z;
}

// Parallax occlusion mapping. The height field is treated as a surface below the top plane:
// depth = 1 - h, so raised texels sit near the top. The view ray enters at the mid-relief plane
// (so the average surface does not drift) and marches through the material in POM_STEPS steps
// until it passes below the height field. The lookup point is linearly interpolated between the
// last sample above the surface and the first sample below it.
const int POM_STEPS = 8;

vec3 pomPoint(sampler2D t, vec3 p, vec3 V, vec3 gn, vec3 w, float tileInv, float depth) {
    vec3 Vt = V - gn * dot(V, gn);
    vec3 dir = Vt / max(dot(V, gn), 0.25) * depth;  // full relief travel, tangent to the surface
    vec3 q = p + 0.5 * dir;                          // start at the top plane
    float r = 0.0;
    float surf = 1.0 - triHeight(t, q, w, tileInv);
    vec3 prevQ = q;
    float prevR = 0.0, prevSurf = surf;
    const float dr = 1.0 / float(POM_STEPS);
    for (int i = 0; i < POM_STEPS; i++) {
        if (r >= surf) break;                        // already below the surface
        prevQ = q; prevR = r; prevSurf = surf;
        r += dr;
        q -= dir * dr;
        surf = 1.0 - triHeight(t, q, w, tileInv);
    }
    if (r < surf) return q;                          // never reached the surface: no occlusion
    // Crossed between prev (in air) and q (in material): interpolate by the depth gaps.
    float gap0 = prevR - prevSurf;                   // negative in air
    float gap1 = r - surf;                           // non-negative in material
    float k = clamp(gap0 / (gap0 - gap1 + 1e-6), 0.0, 1.0);
    return mix(prevQ, q, k);
}

// Baked ambient light at p for normal n, trilinear between the eight surrounding probes.
// Each probe holds ambient cube faces (+x, -x, +y, -y, +z, -z); the result is the cosine-weighted
// average of the sky and bounce light the probe sees along n. False outside the probe grid.
bool probeAmbient(vec3 p, vec3 n, out vec3 ambient) {
    vec3 dims = uF.uProbeDims.xyz;
    vec3 f = (p - uF.uProbeOrigin.xyz) / uF.uProbeStep.xyz;
    if (any(lessThan(f, vec3(0.0))) || any(greaterThan(f, dims - vec3(1.0)))) return false;
    f = min(f, dims - vec3(1.0001));
    ivec3 i0 = ivec3(floor(f));
    vec3 t = f - vec3(i0);
    vec3 an = abs(n);
    float ws = max(an.x + an.y + an.z, 1e-4);
    ambient = vec3(0.0);
    for (int c = 0; c < 8; c++) {
        ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        float w = ((o.x == 1) ? t.x : 1.0 - t.x) * ((o.y == 1) ? t.y : 1.0 - t.y) * ((o.z == 1) ? t.z : 1.0 - t.z);
        int idx = (i0.x + o.x) + int(dims.x) * ((i0.y + o.y) + int(dims.y) * (i0.z + o.z));
        int b = idx * 6;
        vec3 face = an.x * (n.x > 0.0 ? uF.uProbes[b].rgb : uF.uProbes[b + 1].rgb)
                  + an.y * (n.y > 0.0 ? uF.uProbes[b + 2].rgb : uF.uProbes[b + 3].rgb)
                  + an.z * (n.z > 0.0 ? uF.uProbes[b + 4].rgb : uF.uProbes[b + 5].rgb);
        ambient += w * face / ws;
    }
    return true;
}

void evaluateMaterial(out vec3 albedo, out float roughness, out float metallic, out float bump,
                      out vec3 emission, vec3 woodAlb, vec3 rockAlb, out float recess) {
    int kind = int(vMaterial + 0.5);
    vec3 p = vWorld;
    float n0 = noise3(p * 0.42);
    float n1 = noise3(p * 3.7);
    float n2 = noise3(p * 15.0);
    albedo = vAlbedo;
    roughness = 0.72;
    metallic = 0.0;
    bump = 0.025;
    emission = vec3(0.0);
    recess = 0.0;  // facade height field (m): sunk glazing, proud reveals; zero off facades

    if (kind == 0 || kind == 1) {
        // Scanned wood with sun bleaching, embedded dirt, damp end-grain and fine weathering.
        float grain = noise3(vec3(p.x * 17.0, p.y * 2.0, p.z * 17.0));
        float dirt = smoothstep(0.58, 0.88, noise3(p * 1.9 + vec3(8.0, 2.0, 5.0)));
        float bleach = smoothstep(0.25, 0.95, n0) * 0.10;
        float damp = exp(-max(p.y - 0.05, 0.0) * 0.8) * 0.28;
        albedo = woodAlb * (0.91 + 0.13 * n1 + 0.035 * (grain - 0.5));
        albedo = mix(albedo, albedo * vec3(0.66, 0.61, 0.51), dirt * 0.22);
        albedo = mix(albedo, albedo * vec3(1.12, 1.08, 0.98), bleach);
        albedo *= 1.0 - damp * 0.25;
        roughness = kind == 0 ? 0.78 : 0.84;
        roughness = mix(roughness, 0.38, damp);
        bump = 0.0;  // the normal map supplies the relief
    } else if (kind == 2) {
        // Marine paint: broad pigment variation with subtle fine orange-peel texture.
        albedo *= 0.93 + 0.10 * n0 + 0.025 * (n2 - 0.5);
        // Paint chips expose a darker primer and occasional bright, oxidised edges.
        float chip = smoothstep(0.76, 0.94, noise3(p * 13.0 + vec3(3.1, 8.2, 1.7)));
        float edgeWear = pow(1.0 - abs(normalize(vNormal).y), 3.0) * chip;
        albedo = mix(albedo, albedo * vec3(0.30, 0.27, 0.22), chip * 0.42);
        albedo = mix(albedo, vec3(0.34, 0.29, 0.21), edgeWear * 0.34);
        roughness = 0.34 + 0.08 * n1;
        bump = 0.018;
    } else if (kind == 3) {
        // Weathered masonry: broad mineral mottling, porous aggregate, and recessed mortar
        // courses. The courses are deliberately irregular so they do not read as perfect stripes.
        float pores = noise3(p * 11.0);
        float courseNoise = noise3(vec3(p.x * 0.22, p.y * 0.55, p.z * 0.22));
        float course = abs(fract((p.y + (courseNoise - 0.5) * 0.16) * 1.35) - 0.5);
        float mortar = 1.0 - 0.22 * (1.0 - smoothstep(0.025, 0.075, course));
        float mineral = 0.88 + 0.12 * n0 + 0.055 * (n1 - 0.5);
        float pits = 1.0 - 0.07 * smoothstep(0.76, 0.93, pores);
        // Gravity-aligned rain streaks, salt bloom and damp biological staining near the base.
        float streakField = noise3(vec3(p.x * 3.0, p.y * 0.16, p.z * 3.0));
        float streaks = smoothstep(0.57, 0.78, streakField) * (0.35 + 0.65 * n1);
        float salt = smoothstep(0.73, 0.92, noise3(p * 4.2 + vec3(17.0, 0.0, 9.0)));
        float algae = exp(-max(p.y, 0.0) * 0.42) * smoothstep(0.48, 0.78, noise3(p * 1.3));
        albedo *= mineral * pits * mortar;

        // Each tower's palette colour is in vAlbedo, so hashing it gives a stable per-building seed. The seed picks
        // the facade: punched windows (< 0.4), ribbon glazing (< 0.75) or a glass curtain wall (above).
        float seed = fract(sin(dot(vAlbedo, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
        float curtain = step(0.75, seed);
        float wallMask = 1.0 - smoothstep(0.4, 0.8, abs(normalize(vNormal).y));
        vec2 cell = seed < 0.4 ? vec2(3.4, 4.2) : (seed < 0.75 ? vec2(6.0, 3.8) : vec2(1.6, 3.8));
        vec2 glassHalf = seed < 0.4 ? vec2(0.17) : (seed < 0.75 ? vec2(0.40, 0.34) : vec2(0.46, 0.42));

        // Facade grid on vertical faces. wq is the wall coordinate (x + z is constant-normal on axis-aligned walls).
        // Where a cell covers less than a pixel the glazing fades to its average, so distant towers do not shimmer.
        vec2 wq = vec2(p.x + p.z, p.y);
        vec2 cellQ = wq / cell;
        vec2 bayF = abs(fract(cellQ) - 0.5);
        vec2 fw = fwidth(cellQ);
        float aaBay = clamp(max(fw.x, fw.y) * 2.0, 0.0, 1.0);
        float m = max(bayF.x / glassHalf.x, bayF.y / glassHalf.y);  // 1 at the glass edge
        float bay = 1.0 - smoothstep(0.9, 1.1, m);                  // glass pane
        float reveal = smoothstep(1.05, 1.2, m) * (1.0 - smoothstep(1.35, 1.5, m));  // stone surround
        float bayId = noise3(vec3(floor(cellQ.x), floor(cellQ.y), 3.0));
        float share = seed < 0.4 ? smoothstep(0.25, 0.55, bayId) : (seed < 0.75 ? 0.9 : 1.0);
        float avg = seed < 0.4 ? 0.06 : (seed < 0.75 ? 0.35 : 0.80);
        float glazed = wallMask * mix(bay * share, avg, aaBay);

        // Ground-floor shopfront: a dark glazed band under a dark fascia.
        float shop = wallMask * (1.0 - smoothstep(3.6, 4.1, p.y));
        glazed = max(glazed, shop * 0.85);
        float fascia = wallMask * smoothstep(3.9, 4.2, p.y) * (1.0 - smoothstep(4.6, 4.9, p.y));
        albedo = mix(albedo, albedo * 0.45, fascia * 0.8);

        // Curtain wall: the mullions and spandrels between panes are dark anodised aluminium.
        albedo = mix(albedo, vec3(0.10, 0.11, 0.12), curtain * wallMask * (1.0 - bay) * 0.9);
        // Punched and ribbon facades: pale stone reveals around the glazing.
        albedo *= mix(1.0, 1.22, reveal * wallMask * (1.0 - curtain));

        // Glazing: dark glass, and a share of the bays reflect the warm sky and sun.
        float warmGlass = smoothstep(0.55, 0.85, bayId) * 0.6 * (1.0 - curtain);
        vec3 glassTint = mix(vec3(0.10, 0.12, 0.16), vec3(0.45, 0.30, 0.22), warmGlass);
        albedo = mix(albedo, albedo * glassTint + vec3(0.02, 0.018, 0.03), glazed * 0.85);
        albedo = mix(albedo, albedo * vec3(0.53, 0.55, 0.52), streaks * 0.23);
        albedo = mix(albedo, albedo * vec3(1.12, 1.10, 1.02), salt * 0.15);
        albedo = mix(albedo, albedo * vec3(0.38, 0.49, 0.34), algae * 0.30);
        // Each floor band has its own tone; the base is dirtied where rain and spray reach it.
        albedo *= 0.94 + 0.12 * noise3(vec3(0.0, floor(p.y / 4.2), 7.0));
        albedo *= mix(0.78, 1.0, smoothstep(0.0, 14.0, p.y));
        roughness = 0.82 + 0.10 * n2;
        roughness = mix(roughness, 0.48, algae * 0.45);
        roughness = mix(roughness, mix(0.12, 0.08, curtain), glazed * 0.8);
        bump = 0.075;
        // Height field for the normal: glass sunk 6 cm, stone reveals proud 4 cm. Applied in main().
        recess = wallMask * (0.04 * reveal * (1.0 - curtain) - 0.06 * glazed);
    } else if (kind == 4) {
        // Dark galvanised metal: fine directional-looking grain and restrained oxidation.
        float grain = noise3(vec3(p.x * 38.0, p.y * 5.0, p.z * 38.0));
        float oxidation = smoothstep(0.68, 0.91, noise3(p * 2.4));
        float scratches = smoothstep(0.72, 0.91, noise3(vec3(p.x * 55.0, p.y * 8.0, p.z * 55.0)));
        float edge = pow(1.0 - abs(normalize(vNormal).y), 2.0);
        albedo *= (0.91 + 0.09 * n1) * (1.0 - 0.12 * oxidation);
        albedo = mix(albedo, vec3(0.23, 0.16, 0.10), oxidation * 0.28);
        albedo = mix(albedo, vec3(0.48, 0.46, 0.40), scratches * edge * 0.18);
        roughness = 0.25 + 0.16 * mix(n2, grain, 0.65);
        roughness = mix(roughness, 0.72, oxidation * 0.35);
        metallic = 0.68;
        bump = 0.018;
    } else if (kind == 5) {
        // Warm lantern glazing and beacon core. The visible warm emission is deliberately
        // localised; it is not a replacement for a proper light-volume pass.
        albedo *= 0.90 + 0.10 * n1;
        roughness = 0.12;
        emission = vec3(3600.0, 1550.0, 390.0) * uF.uExposure.x;
    } else if (kind == 6) {
        // Scanned coastal rock with a procedural mineral tint and fracture darkening.
        float fracture = noise3(p * 7.5 + vec3(2.3, 0.0, 4.1));
        albedo = rockAlb * (0.85 + 0.20 * n0) * (1.0 - 0.10 * smoothstep(0.70, 0.91, fracture));
        // Wet band at the waterline: darker and glossier, fading over a few metres.
        float wet = exp(-max(p.y - 0.4, 0.0) * 0.45);
        float lichen = smoothstep(0.60, 0.86, noise3(p * 2.1 + vec3(4.0, 9.0, 1.0))) *
                       (0.35 + 0.65 * noise3(p * 0.45));
        albedo *= mix(1.0, 0.6, wet);
        albedo = mix(albedo, albedo * vec3(0.58, 0.68, 0.43), lichen * 0.30);
        roughness = mix(0.91 + 0.045 * n2, 0.30, wet * 0.72);
        bump = 0.0;  // the normal map supplies the relief
    } else if (kind == 7) {
        // Fine-grained beach sand, slightly compacted and darkened close to the waterline.
        float grain = noise3(p * 34.0) * 0.55 + noise3(p * 11.0) * 0.45;
        float damp = exp(-abs(p.y - 0.12) * 0.72);
        albedo *= (0.88 + 0.16 * grain) * mix(1.0, 0.67, damp * 0.72);
        roughness = mix(0.97, 0.72, damp * 0.65);
        bump = 0.055;
    } else if (kind == 8) {
        // Asphalt: dark aggregate with patched repairs. Tyres use the same branch with a near-black vAlbedo.
        float agg = noise3(p * 26.0) * 0.6 + noise3(p * 7.0) * 0.4;
        float repair = smoothstep(0.66, 0.84, noise3(p * 0.9 + vec3(5.0, 1.0, 2.0)));
        albedo = vAlbedo * (0.82 + 0.36 * agg) * mix(1.0, 1.35, repair * 0.5);
        roughness = 0.88 + 0.06 * n2;
        bump = 0.03;
    } else if (kind == 9) {
        // Road paint: slightly scuffed, with a faint tyre-dust tint.
        albedo = vAlbedo * (0.90 + 0.10 * n1) * (1.0 - 0.10 * smoothstep(0.6, 0.9, n0));
        roughness = 0.45 + 0.08 * n2;
        bump = 0.0;
    } else if (kind == 10) {
        // Pavers: 0.6 m flags with darker mortar joints and a small tone variation per flag.
        vec2 g = fract(p.xz / 0.6);
        vec2 gd = min(g, 1.0 - g);
        float joint = 1.0 - smoothstep(0.012, 0.03, min(gd.x, gd.y));
        float flagTone = 0.88 + 0.22 * hash31(vec3(floor(p.xz / 0.6), 4.0));
        albedo = vAlbedo * flagTone * (0.94 + 0.10 * n1);
        albedo = mix(albedo, albedo * 0.55, joint);
        roughness = 0.78 + 0.08 * n2;
        bump = 0.02;
    } else if (kind == 11) {
        // Car paint: a clearcoat over the body colour, with faint orange peel.
        albedo = vAlbedo * (0.96 + 0.06 * n2);
        roughness = 0.20 + 0.05 * n2;
        bump = 0.004;
    } else if (kind == 12) {
        // Dark glazing on road vehicles: nearly smooth, so it reflects the sky.
        albedo = vAlbedo * (0.8 + 0.4 * n1);
        roughness = 0.06;
        bump = 0.0;
    } else if (kind == 13) {
        // Stone trim (cornices, pilasters): light mottling, with grime gathering low down.
        float grime = smoothstep(0.55, 0.9, n1) * exp(-max(p.y - 1.0, 0.0) * 0.25);
        albedo = vAlbedo * (0.92 + 0.14 * n0) * mix(1.0, 0.86, grime);
        roughness = 0.74 + 0.08 * n2;
        bump = 0.03;
    }

    // Salt spray and persistent moisture affect exposed coastal surfaces. This is deliberately
    // world-space and low frequency so it does not crawl with camera movement.
    float coastalWet = exp(-abs(p.y - 0.08) * 1.15) * smoothstep(0.18, 0.82, noise3(p * 0.31));
    if (kind != 5) {
        roughness = mix(roughness, max(0.16, roughness * 0.48), coastalWet * 0.42);
        albedo *= 1.0 - coastalWet * 0.12;
    }

    // --- Weather-dependent wetness, rain, and submersion effects ---
    float rain = uF.uWeather.x;
    float wet = uF.uWeather.y;
    float flood = uF.uWeather.z;

    // Water submersion: when the surface is below flood level, darken and desaturate
    float submerged = step(flood, p.y) * smoothstep(-10.0, 0.0, p.y - flood);
    float submix = max(0.0, min(1.0, submerged));
    // Darken submerged surfaces (decrease albedo, increase darkness)
    albedo *= 1.0 - submix * 0.35;
    // Reduce roughness when submerged (darker, "deeper" water look)
    roughness = mix(roughness, max(0.1, roughness * 0.6), submix * 0.5);

    // Rain streaks on vertical/near-vertical faces: apply only when raining
    float rainStreaks = 0.0;
    if (rain > 0.01) {
        // Rain direction is roughly -y (downwards); project onto vertical faces
        float verticality = abs(dot(normalize(vNormal), vec3(0.0, 1.0, 0.0)));
        if (verticality > 0.3) {
            rainStreaks = rain * 0.5 * (1.0 - verticality); // more streaks on vertical faces
            // Slightly darken where rain streaks hit
            albedo *= 1.0 - rainStreaks * 0.15;
            // Increase roughness slightly where rain hits
            roughness = mix(roughness, roughness + 0.03 * rainStreaks, 0.5);
        }
    }

    // Wet surface response: when wet, surfaces get darker and more reflective
    // Wetness factor y in uWeather controls the overall wetness level
    if (wet > 0.01) {
        // Wet areas darken (decrease albedo) and get a slight blue tint
        vec3 wetTint = vec3(0.95, 0.97, 1.0) * wet * 0.15;
        albedo = mix(albedo, albedo * wetTint, 0.3);
        // Wet surfaces have lower roughness (smoother, more mirror-like)
        roughness = mix(roughness, max(0.05, roughness * 0.7), wet * 0.4);
        // Add a slight sheen/gloss to simulate the oil film on water
        // This is handled in the BRDF, not directly in albedo/roughness
    }

    // Terrain material blending based on moisture: sand becomes darker/glossier near water,
    // rock develops algae/moss growth
    if (kind == 7 || kind == 6) {  // sand or rock
        // Flood level affects how much of the material is "underwater"
        float floodFactor = smoothstep(flood - 2.0, flood, p.y);
        // Decrease albedo and increase darkness underwater
        albedo *= 1.0 - floodFactor * 0.25;
        // Wet sand/rock gets slightly smoother (lower roughness)
        roughness = mix(roughness, max(0.05, roughness - 0.05 * floodFactor), 0.3);
        // Sand develops moisture-dependent micro-sparkle when wet
        if (kind == 7 && wet > 0.01) {
            float moistureSparkle = smoothstep(0.0, 0.3, wet) * smoothstep(0.0, 0.5, floodFactor);
            albedo *= 1.0 + vec3(0.02, 0.01, 0.0) * moistureSparkle;  // golden sparkle
            roughness = mix(roughness, max(0.02, roughness - 0.02 * moistureSparkle), 0.4);
        }
        // Rock develops algae/moss growth in damp shaded areas
        if (kind == 6 && wet > 0.01) {
            float algae = smoothstep(0.4, 0.8, noise3(p * 3.0)) * smoothstep(0.0, 0.6, floodFactor);
            vec3 algaeTint = vec3(0.2, 0.35, 0.15) * algae;
            albedo = mix(albedo, albedo * algaeTint, 0.3);
            roughness = mix(roughness, roughness + 0.02 * algae, 0.3);
        }
    }

    // Sub-pixel material breakup: real surfaces do not have one perfectly uniform roughness.
    // Keep the amplitude low enough to avoid noisy highlights at mobile resolutions.
    float micro = noise3(p * 46.0);
    float meso = noise3(p * 7.5 + vec3(13.1, 2.7, 8.4));
    roughness = clamp(roughness + (micro - 0.5) * 0.055 + (meso - 0.5) * 0.075,
                      0.045, 0.985);
    albedo *= 0.975 + 0.05 * micro;
}

// Screen-space reflection: march the reflected ray through this frame's linear depth. A hit is a
// sample that lies just behind the depth buffer; its colour is last frame's resolved HDR (uHistory).
// Returns rgb and coverage (0 on a miss or at the screen edge).
vec4 ssrTrace(vec3 p, vec3 R, vec3 cam) {
    if (uF.uTaa.z < 0.5) return vec4(0.0);
    const int SSR_STEPS = 24;
    const float SSR_MAX = 60.0;
    for (int i = 0; i < SSR_STEPS; i++) {
        float f = float(i + 1) / float(SSR_STEPS);
        vec3 q = p + R * (0.15 + SSR_MAX * f * f);
        vec4 c = uF.uMainViewProj * vec4(q, 1.0);
        if (c.w <= 0.0) break;
        vec2 suv = c.xy / c.w * 0.5 + 0.5;
        if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) break;
        float sceneD = texture(uLinDepth, suv).r;
        if (sceneD <= 0.0) continue;
        float rayD = distance(q, cam);
        float thick = 0.5 + 0.03 * rayD;
        if (rayD > sceneD && rayD < sceneD + thick) {
            float edge = min(min(suv.x, suv.y), min(1.0 - suv.x, 1.0 - suv.y));
            return vec4(texture(uHistory, suv).rgb, smoothstep(0.0, 0.08, edge));
        }
    }
    return vec4(0.0);
}

// Screen-space surface gradient (Mikkelsen 2010): perturbs N by a height field h in metres, with no tangent frame.
// Zero height returns N unchanged. Must run at top level in main(), where the screen derivatives are defined.
vec3 perturbNormal(vec3 N, float h) {
    vec3 dpdx = dFdx(vWorld), dpdy = dFdy(vWorld);
    float dhdx = dFdx(h), dhdy = dFdy(h);
    vec3 r1 = cross(dpdy, N), r2 = cross(N, dpdx);
    float det = dot(dpdx, r1);
    vec3 surfGrad = sign(det) * (dhdx * r1 + dhdy * r2);
    vec3 r = abs(det) * N - surfGrad;
    return dot(r, r) > 1e-20 ? normalize(r) : N;
}

void main() {
    // The planar reflection is rendered from a camera mirrored in the sea: only the part of the
    // scene above the surface reflects, so geometry below it is dropped there.
    if (uF.uProbeDims.w > 0.5 && vWorld.y < -0.02) discard;

    vec3 gn = normalize(vNormal);
    vec3 w = triWeights(gn);
    vec3 cam = uF.uCamPos.xyz;
    vec3 V = normalize(cam - vWorld);

    // Parallax first: the height is blended from the dominant projections, then every texture
    // lookup for that material uses the shifted point. Texture samples are unconditional, so mip
    // selection is well defined where neighbouring pixels use different materials.
    vec3 pW = pomPoint(uWoodNormal, vWorld, V, gn, w, 1.0 / WOOD_TILE, WOOD_DEPTH);
    vec3 pR = pomPoint(uRockNormal, vWorld, V, gn, w, 1.0 / ROCK_TILE, ROCK_DEPTH);
    vec3 woodAlb = WOOD_GAIN * pow(triAlbedo(uWoodAlbedo, pW, w, 1.0 / WOOD_TILE), vec3(2.2));
    vec3 rockAlb = ROCK_GAIN * pow(triAlbedo(uRockAlbedo, pR, w, 1.0 / ROCK_TILE), vec3(2.2));
    vec3 woodN = triNormal(uWoodNormal, pW, gn, w, 1.0 / WOOD_TILE);
    vec3 rockN = triNormal(uRockNormal, pR, gn, w, 1.0 / ROCK_TILE);

    vec3 albedo;
    float roughness, metallic, bump;
    vec3 emission;
    float recess;
    evaluateMaterial(albedo, roughness, metallic, bump, emission, woodAlb, rockAlb, recess);

    int kind = int(vMaterial + 0.5);
    vec3 n = gn;
    // Facade recess: derivatives are taken here, in uniform control flow, not inside the material branches.
    vec3 nRecess = perturbNormal(gn, recess);
    if (kind == 0 || kind == 1) {
        n = woodN;
    } else if (kind == 6) {
        n = rockN;
    } else {
        // A small derivative-free procedural bump adds surface breakup without a normal-map asset.
        vec3 e = vec3(0.035, 0.0, 0.0);
        vec3 grad = vec3(
            noise3(vWorld + e) - noise3(vWorld - e),
            noise3(vWorld + e.yxz) - noise3(vWorld - e.yxz),
            noise3(vWorld + e.zyx) - noise3(vWorld - e.zyx));
        vec3 eFine = e * 0.22;
        grad += 0.28 * vec3(
            noise3(vWorld * 4.0 + eFine) - noise3(vWorld * 4.0 - eFine),
            noise3(vWorld * 4.0 + eFine.yxz) - noise3(vWorld * 4.0 - eFine.yxz),
            noise3(vWorld * 4.0 + eFine.zyx) - noise3(vWorld * 4.0 - eFine.zyx));
        grad -= gn * dot(grad, gn);
        n = normalize(gn - bump * grad * 2.6);
    }
    if (kind == 3) n = normalize(n + nRecess - gn);

    float NoV = max(dot(n, V), 1e-4);
    vec3 S = normalize(uF.uSunDir.xyz);
    float NoL = max(dot(n, S), 0.0);
    float alpha = max(roughness * roughness, 0.018);
    vec3 F0 = mix(vec3(DIELECTRIC_F0), albedo, metallic);
    vec3 Fv = fresnelSchlick(F0, NoV);
    vec3 diffuseColor = albedo * (1.0 - metallic) * (1.0 - Fv);
    float visibility = NoL > 0.0
        ? sunVisibility(vWorld, n) * cloudShadowTransmittance(vWorld) : 0.0;
    vec3 directDiffuse = (diffuseColor / PI) * uF.uSunIlluminance.xyz * NoL * visibility;

    vec3 directSpecular = vec3(0.0);
    if (NoL > 0.0) {
        vec3 H = normalize(V + S);
        float NoH = max(dot(n, H), 0.0);
        float VoH = max(dot(V, H), 1e-4);
        float D = distributionGGX(NoH, alpha);
        float G = smithG1(NoV, alpha) * smithG1(NoL, alpha);
        vec3 F = fresnelSchlick(F0, VoH);
        vec3 brdf = D * G * F / max(4.0 * NoV * NoL, 1e-5);
        directSpecular = brdf * uF.uSunIlluminance.xyz * NoL * visibility;
    }

    // Ambient: baked probes (sky occlusion and bounce) where the grid covers the point, the sky
    // LUT elsewhere. Reflections are the sky blurred over a cone that widens with roughness.
    vec3 probeLight = vec3(0.0);
    bool probed = probeAmbient(vWorld, n, probeLight);
    vec3 ambientDiffuse = probed ? probeLight : skyDiffuseIrradiance(n);
    vec3 R = reflect(-V, n);
    vec3 envRadiance = skyGlossy(vec3(R.x, max(R.y, 0.0005), R.z), 0.02 + 0.5 * roughness * roughness);
    // Only glossy surfaces take screen-space reflections; rough ones keep the sky.
    vec4 ssr = ssrTrace(vWorld + n * 0.05, R, cam);
    envRadiance = mix(envRadiance, ssr.rgb, ssr.a * (1.0 - smoothstep(0.35, 0.7, roughness)));
    // A compact multiple-scattering compensation keeps rough conductors/dielectrics from
    // becoming unnaturally dark as the microfacet lobe broadens (approximation to the LUT method).
    float energyLoss = 1.0 - 0.28 * roughness * (1.0 - max(max(F0.r, F0.g), F0.b));
    vec3 envSpecular = envRadiance * Fv / max(energyLoss, 0.55);
    float ao = clamp(0.72 + 0.28 * noise3(vWorld * 1.8 + gn * 0.7), 0.55, 1.0);
    ambientDiffuse *= ao;
    // Screen-space AO from the main camera (the mirrored pass samples the same main-view texels).
    vec4 cAO = uF.uMainViewProj * vec4(vWorld, 1.0);
    float ssao = texture(uAO, cAO.xy / cAO.w * 0.5 + 0.5).r;
    ambientDiffuse *= ssao;

    float dist = distance(vWorld, cam);
    vec3 Tv;
    vec3 Lin = aerialLookup(vWorld, dist, Tv);
    vec3 radiance = directDiffuse + directSpecular + diffuseColor * ambientDiffuse + envSpecular + emission;
    outHdr = vec4(radiance * Tv + Lin, 1.0);
}
