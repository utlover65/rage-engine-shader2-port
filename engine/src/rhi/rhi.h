// Backend-agnostic rendering hardware interface. Nothing above src/rhi includes Vulkan
// headers. Errors are reported by throwing rhi::Error (the engine has no recoverable
// graphics errors: a failed device call means the frame cannot be produced).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine {
namespace rhi {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class Backend : uint8_t { Vulkan };

enum class Format : uint8_t {
    RGBA8_UNORM,   // display output (sRGB encode is done in the shader)
    RGBA16_FLOAT,  // HDR radiance and lookup tables
    R32_FLOAT,     // single-float maps (sun shadow depth)
    D32_FLOAT,     // depth
};

// Where an image may be used. The render graph moves images between these; a backend
// translates them into its own layouts and barriers.
enum class ImageState : uint8_t {
    Undefined,
    ColorTarget,
    DepthTarget,
    ShaderRead,
    TransferSrc,
    TransferDst,
};

enum class LoadOp : uint8_t { Clear, Load };

// Staging: source of an upload copy (CPU writes, GPU reads).
enum class BufferUsage : uint8_t { Vertex, Index, Uniform, Readback, Staging, Storage };

enum class VertexFormat : uint8_t { Float1, Float2, Float3, Float4 };

struct ImageDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    Format format = Format::RGBA16_FLOAT;
    bool colorTarget = false;  // may be rendered to as a colour attachment
    bool depthTarget = false;  // may be rendered to as a depth attachment
    bool sampled = false;      // may be read by shaders
    bool transferSrc = false;  // may be copied to a buffer
    std::string name;
    uint32_t mipLevels = 1;    // 1 for render targets; textures may carry a full chain
};

struct VertexAttribute {
    uint32_t location = 0;
    uint32_t offset = 0;
    VertexFormat format = VertexFormat::Float2;
};

struct PipelineDesc {
    std::string name;
    const class Shader* vertex = nullptr;
    const class Shader* fragment = nullptr;
    Format colorFormat = Format::RGBA16_FLOAT;
    bool hasDepth = false;
    bool depthTest = false;
    bool depthWrite = false;
    uint32_t vertexStride = 0;                    // 0 = no vertex input (shader generates vertices)
    std::vector<VertexAttribute> vertexAttributes;
    uint32_t pushConstantBytes = 0;               // at most 128
    bool cullBack = false;
};

struct ComputePipelineDesc {
    std::string name;
    const class Shader* compute = nullptr;
    uint32_t pushConstantBytes = 0;  // at most 128
};

class Image {
public:
    virtual ~Image() = default;
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
    virtual Format format() const = 0;
    virtual const std::string& name() const = 0;
};

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    // Host-visible storage, mapped for the buffer's whole lifetime. Writes are visible to
    // the device at the next submit; reads after submit() returns are valid.
    virtual void* data() = 0;
};

class Shader {
public:
    virtual ~Shader() = default;
};

class Pipeline {
public:
    virtual ~Pipeline() = default;
};

// Binding layout shared by every pipeline (fixed by the device):
//   binding 0          uniform buffer (vertex and fragment stages)
//   bindings 1..13     combined image sampler (fragment stage), linear filtering with
//                      mip interpolation, clamp addressing (shaders wrap by fract() if needed)
class DescriptorSet {
public:
    virtual ~DescriptorSet() = default;
};

constexpr uint32_t kMaxSampledImages = 22;
// Storage-buffer slots per descriptor set. Slots 0 and 1 are compute in and out; slots 2 and 3 are read-only in the
// vertex stage (ocean displacement fields, slots 0..2 of the grid) and in the fragment stage (slot 3, the capillary
// slopes). Read-only storage needs no device feature.
constexpr uint32_t kMaxStorageBuffers = 4;
// GPU timestamp queries: the query pool holds this many (see CommandList::writeTimestamp).
constexpr uint32_t kMaxTimestamps = 256;

class CommandList {
public:
    virtual ~CommandList() = default;

    virtual void begin() = 0;
    virtual void end() = 0;

    // Moves an image to the state a later command needs.
    virtual void transition(Image& image, ImageState to) = 0;

    // Opens a rendering scope on the targets. Viewport and scissor cover the colour target.
    // depth may be null. Targets must already be in ColorTarget / DepthTarget states.
    virtual void beginRendering(Image& color, LoadOp colorLoad, const std::array<float, 4>& clearColor,
                                Image* depth) = 0;
    virtual void endRendering() = 0;

    virtual void bindPipeline(const Pipeline& pipeline) = 0;
    virtual void bindDescriptorSet(const DescriptorSet& set) = 0;
    virtual void bindVertexBuffer(const Buffer& buffer) = 0;
    virtual void bindIndexBuffer(const Buffer& buffer) = 0;  // 32-bit indices
    virtual void pushConstants(const void* data, uint32_t bytes) = 0;

    virtual void draw(uint32_t vertexCount) = 0;
    virtual void drawIndexed(uint32_t indexCount) = 0;

    // Copies a whole image (must be in TransferSrc) into a buffer of at least
    // width*height*bytesPerPixel bytes, tightly packed, row 0 first.
    virtual void copyImageToBuffer(Image& image, Buffer& buffer) = 0;

    // Uploads every mip level of an image (must be in TransferDst). The buffer holds the
    // levels tightly packed, level 0 first; level k is max(1, w >> k) by max(1, h >> k)
    // texels, each texel 4 bytes for RGBA8.
    virtual void copyBufferToImage(Buffer& buffer, Image& image) = 0;

    // Compute: bind a compute pipeline, its descriptor set and push constants, then dispatch
    // x * y * z workgroups. The storage buffer lives at binding kMaxSampledImages + 1.
    virtual void bindComputePipeline(const Pipeline& pipeline) = 0;
    virtual void dispatch(uint32_t x, uint32_t y, uint32_t z) = 0;
    // Orders shader writes before shader reads between dispatches (compute to compute and compute to vertex).
    virtual void computeBarrier() = 0;

    // GPU timestamps: reset the first count queries, then write one at this point of the list.
    virtual void resetTimestamps(uint32_t count) = 0;
    virtual void writeTimestamp(uint32_t index) = 0;
};

class Device {
public:
    virtual ~Device() = default;

    virtual const std::string& deviceName() const = 0;
    virtual const char* backendName() const = 0;
    // Validation-layer errors reported by the driver stack (0 when validation is off).
    virtual uint32_t validationErrorCount() const = 0;

    virtual std::unique_ptr<Image> createImage(const ImageDesc& desc) = 0;
    virtual std::unique_ptr<Buffer> createBuffer(size_t bytes, BufferUsage usage) = 0;
    virtual std::unique_ptr<Shader> createShader(const std::string& spirvPath) = 0;
    virtual std::unique_ptr<Pipeline> createPipeline(const PipelineDesc& desc) = 0;
    virtual std::unique_ptr<Pipeline> createComputePipeline(const ComputePipelineDesc& desc) = 0;
    // Binds a read/write storage buffer into a descriptor set (compute shaders).
    // slot 0 = binding kMaxSampledImages + 1 (input), slot 1 = + 2 (output), slot 2 = + 3 (vertex-readable).
    virtual void setStorageBuffer(DescriptorSet& set, Buffer& buffer, uint32_t slot) = 0;
    virtual std::unique_ptr<DescriptorSet> createDescriptorSet() = 0;
    // uniform may be null if the pipeline does not read it. Unused texture slots are ignored.
    virtual void updateDescriptorSet(DescriptorSet& set, const Buffer* uniform,
                                     const std::array<const Image*, kMaxSampledImages>& textures) = 0;

    virtual std::unique_ptr<CommandList> createCommandList() = 0;
    // Submits the list and blocks until the GPU has finished it.
    virtual void submitAndWait(CommandList& list) = 0;

    // Ticks written by the last submitted list (call after submitAndWait). Multiply by timestampPeriodNs() for
    // nanoseconds. readTimestamps returns nothing and timestampsSupported is false when the device has no timers.
    virtual std::vector<uint64_t> readTimestamps(uint32_t count) = 0;
    virtual double timestampPeriodNs() const = 0;
    virtual bool timestampsSupported() const = 0;
};

struct DeviceOptions {
    bool validation = false;  // enable the Khronos validation layer and report its messages
};

std::unique_ptr<Device> createDevice(Backend backend, const DeviceOptions& options);

}  // namespace rhi
}  // namespace engine
