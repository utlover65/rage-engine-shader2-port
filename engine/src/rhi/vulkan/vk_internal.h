// Vulkan implementation of rhi.h. Internal to the backend: include only from
// src/rhi/vulkan/*.cpp.
#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

#include "rhi/rhi.h"

namespace engine {
namespace rhi {
namespace vk {

// Throws rhi::Error when a Vulkan call fails.
#define VK_CHECK(call, what)                                                        \
    do {                                                                            \
        VkResult vkCheckResult_ = (call);                                           \
        if (vkCheckResult_ != VK_SUCCESS)                                           \
            throw ::engine::rhi::Error(std::string(what) + " failed: VkResult " +  \
                                       std::to_string(static_cast<int>(vkCheckResult_))); \
    } while (0)

// Device-wide state shared by all resources and command lists.
struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    uint32_t validationErrors = 0;
    VkQueryPool timestampPool = VK_NULL_HANDLE;  // GPU timestamps (kMaxTimestamps queries)
    bool timestamps = false;                     // the device has timers for graphics and compute
    float timestampPeriod = 0.0f;                // nanoseconds per tick

    uint32_t findMemory(uint32_t typeBits, VkMemoryPropertyFlags props) const;
};

// Maps the RHI formats to Vulkan.
VkFormat toVkFormat(Format f);
// Image aspect for a format.
VkImageAspectFlags aspectOf(Format f);
// Layout a layout-tracked image must be in for a given state.
VkImageLayout layoutFor(ImageState s);

class ImageImpl final : public Image {
public:
    ImageImpl(Context& ctx, const ImageDesc& desc);
    ~ImageImpl() override;
    uint32_t width() const override { return width_; }
    uint32_t height() const override { return height_; }
    Format format() const override { return format_; }
    const std::string& name() const override { return name_; }

    Context& ctx;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;  // tracked; barriers move it
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t mipLevels = 1;

private:
    uint32_t width_, height_;
    Format format_;
    std::string name_;
};

class BufferImpl final : public Buffer {
public:
    BufferImpl(Context& ctx, size_t bytes, BufferUsage usage);
    ~BufferImpl() override;
    size_t size() const override { return size_; }
    void* data() override { return mapped; }

    Context& ctx;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;

private:
    size_t size_;
};

class ShaderImpl final : public Shader {
public:
    ShaderImpl(Context& ctx, const std::string& spirvPath);
    ~ShaderImpl() override;
    Context& ctx;
    VkShaderModule module = VK_NULL_HANDLE;
};

class PipelineImpl final : public Pipeline {
public:
    PipelineImpl(Context& ctx, const PipelineDesc& desc);
    PipelineImpl(Context& ctx, const ComputePipelineDesc& desc);
    ~PipelineImpl() override;
    Context& ctx;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t pushBytes = 0;
    bool compute = false;
};

class DescriptorSetImpl final : public DescriptorSet {
public:
    explicit DescriptorSetImpl(Context& ctx);
    ~DescriptorSetImpl() override;
    Context& ctx;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

class CommandListImpl final : public CommandList {
public:
    explicit CommandListImpl(Context& ctx);
    ~CommandListImpl() override;

    void begin() override;
    void end() override;
    void transition(Image& image, ImageState to) override;
    void beginRendering(Image& color, LoadOp colorLoad, const std::array<float, 4>& clearColor,
                        Image* depth) override;
    void endRendering() override;
    void bindPipeline(const Pipeline& pipeline) override;
    void bindDescriptorSet(const DescriptorSet& set) override;
    void bindComputePipeline(const Pipeline& pipeline) override;
    void dispatch(uint32_t x, uint32_t y, uint32_t z) override;
    void computeBarrier() override;
    void bindVertexBuffer(const Buffer& buffer) override;
    void bindIndexBuffer(const Buffer& buffer) override;
    void pushConstants(const void* data, uint32_t bytes) override;
    void draw(uint32_t vertexCount) override;
    void drawIndexed(uint32_t indexCount) override;
    void copyImageToBuffer(Image& image, Buffer& buffer) override;
    void copyBufferToImage(Buffer& buffer, Image& image) override;
    void resetTimestamps(uint32_t count) override;
    void writeTimestamp(uint32_t index) override;

    Context& ctx;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    const PipelineImpl* boundPipeline = nullptr;
    const DescriptorSetImpl* boundSet = nullptr;
};

class DeviceImpl final : public Device {
public:
    explicit DeviceImpl(const DeviceOptions& options);
    ~DeviceImpl() override;

    const std::string& deviceName() const override { return name_; }
    const char* backendName() const override { return "Vulkan"; }
    uint32_t validationErrorCount() const override { return ctx.validationErrors; }

    std::unique_ptr<Image> createImage(const ImageDesc& desc) override;
    std::unique_ptr<Buffer> createBuffer(size_t bytes, BufferUsage usage) override;
    std::unique_ptr<Shader> createShader(const std::string& spirvPath) override;
    std::unique_ptr<Pipeline> createPipeline(const PipelineDesc& desc) override;
    std::unique_ptr<Pipeline> createComputePipeline(const ComputePipelineDesc& desc) override;
    void setStorageBuffer(DescriptorSet& set, Buffer& buffer, uint32_t slot) override;
    std::unique_ptr<DescriptorSet> createDescriptorSet() override;
    void updateDescriptorSet(DescriptorSet& set, const Buffer* uniform,
                             const std::array<const Image*, kMaxSampledImages>& textures) override;
    std::unique_ptr<CommandList> createCommandList() override;
    void submitAndWait(CommandList& list) override;
    std::vector<uint64_t> readTimestamps(uint32_t count) override;
    double timestampPeriodNs() const override;
    bool timestampsSupported() const override;

    Context ctx;

private:
    std::string name_;
};

}  // namespace vk
}  // namespace rhi
}  // namespace engine
