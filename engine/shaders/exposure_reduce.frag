// Auto-exposure: final reduction. Average the luminance grid into a single value (the target is
// 1x1). The grid is 64x64, so a coarse 8x8 tap pattern is a good average.
layout(location = 0) out float outLum;

void main() {
    float sum = 0.0;
    int n = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            vec2 p = (vec2(float(x), float(y)) + 0.5) / 8.0;
            sum += log(max(texture(uHdrTex, p).r, 1e-4));
            n++;
        }
    outLum = exp(sum / float(n));
}
