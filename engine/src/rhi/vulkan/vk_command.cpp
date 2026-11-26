// Vulkan command recording. Barriers are coarse (all commands, all memory) on purpose:
// correct first, narrowed later once the frame is profiled on the target device.
#include <algorithm>

#include "rhi/vulkan/vk_internal.h"

namespace engine {
namespace rhi {
namespace vk {

CommandListImpl::CommandListImpl(Context& c) : ctx(c) {
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = ctx.cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai, &cb), "vkAllocateCommandBuffers");
}

CommandListImpl::~CommandListImpl() { vkFreeCommandBuffers(ctx.device, ctx.cmdPool, 1, &cb); }

void CommandListImpl::begin() {
    boundPipeline = nullptr;
    VK_CHECK(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
}

void CommandListImpl::end() { VK_CHECK(vkEndCommandBuffer(cb), "vkEndCommandBuffer"); }

void CommandListImpl::resetTimestamps(uint32_t count) {
    if (!ctx.timestamps) return;
    if (count > kMaxTimestamps) throw Error("too many timestamps requested");
    vkCmdResetQueryPool(cb, ctx.timestampPool, 0, count);
}

void CommandListImpl::writeTimestamp(uint32_t index) {
    if (!ctx.timestamps) return;
    if (index >= kMaxTimestamps) throw Error("timestamp index out of range");
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, ctx.timestampPool, index);
}

void CommandListImpl::transition(Image& imageBase, ImageState to) {
    auto& image = static_cast<ImageImpl&>(imageBase);
    const VkImageLayout target = layoutFor(to);
    if (image.layout == target) return;

    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = image.layout;
    b.newLayout = target;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image.image;
    b.subresourceRange = {image.aspect, 0, image.mipLevels, 0, 1};
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    image.layout = target;
}

void CommandListImpl::beginRendering(Image& colorBase, LoadOp colorLoad, const std::array<float, 4>& clear,
                                     Image* depthBase) {
    auto& color = static_cast<ImageImpl&>(colorBase);
    if (color.layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        throw Error("colour target '" + color.name() + "' is not in ColorTarget state");

    VkRenderingAttachmentInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ca.imageView = color.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = colorLoad == LoadOp::Clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color.float32[0] = clear[0];
    ca.clearValue.color.float32[1] = clear[1];
    ca.clearValue.color.float32[2] = clear[2];
    ca.clearValue.color.float32[3] = clear[3];

    VkRenderingAttachmentInfo da{};
    if (depthBase) {
        auto& depth = static_cast<ImageImpl&>(*depthBase);
        if (depth.layout != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
            throw Error("depth target '" + depth.name() + "' is not in DepthTarget state");
        if (depth.width() != color.width() || depth.height() != color.height())
            throw Error("depth and colour targets differ in size");
        da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        da.imageView = depth.view;
        da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;  // depth is cleared at the start of every scope
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        da.clearValue.depthStencil = {1.0f, 0};
    }

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {color.width(), color.height()}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = depthBase ? &da : nullptr;
    vkCmdBeginRendering(cb, &ri);

    VkViewport vp{0.0f, 0.0f, static_cast<float>(color.width()), static_cast<float>(color.height()), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {color.width(), color.height()}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
}

void CommandListImpl::endRendering() { vkCmdEndRendering(cb); }

void CommandListImpl::bindPipeline(const Pipeline& pipelineBase) {
    boundPipeline = static_cast<const PipelineImpl*>(&pipelineBase);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, boundPipeline->pipeline);
}

void CommandListImpl::bindDescriptorSet(const DescriptorSet& setBase) {
    if (!boundPipeline) throw Error("bindDescriptorSet before bindPipeline");
    const auto& set = static_cast<const DescriptorSetImpl&>(setBase);
    boundSet = &set;  // remembered so dispatch can bind it at the compute point
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, boundPipeline->layout, 0, 1, &set.set, 0, nullptr);
}

void CommandListImpl::bindComputePipeline(const Pipeline& pipelineBase) {
    const auto* pipeline = static_cast<const PipelineImpl*>(&pipelineBase);
    if (!pipeline->compute) throw Error("bindComputePipeline with a graphics pipeline");
    boundPipeline = pipeline;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
}

void CommandListImpl::computeBarrier() {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
}

void CommandListImpl::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    if (!boundPipeline || !boundPipeline->compute) throw Error("dispatch without a compute pipeline");
    if (boundSet) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, boundPipeline->layout, 0, 1, &boundSet->set, 0,
                                nullptr);
    }
    vkCmdDispatch(cb, x, y, z);
}

void CommandListImpl::bindVertexBuffer(const Buffer& bufferBase) {
    const auto& buffer = static_cast<const BufferImpl&>(bufferBase);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &buffer.buffer, &offset);
}

void CommandListImpl::bindIndexBuffer(const Buffer& bufferBase) {
    const auto& buffer = static_cast<const BufferImpl&>(bufferBase);
    vkCmdBindIndexBuffer(cb, buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
}

void CommandListImpl::pushConstants(const void* data, uint32_t bytes) {
    if (!boundPipeline || bytes > boundPipeline->pushBytes) throw Error("push constant size mismatch");
    const VkShaderStageFlags stages = boundPipeline->compute
                                          ? VK_SHADER_STAGE_COMPUTE_BIT
                                          : (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    vkCmdPushConstants(cb, boundPipeline->layout, stages, 0, bytes,
                       data);
}

void CommandListImpl::draw(uint32_t vertexCount) { vkCmdDraw(cb, vertexCount, 1, 0, 0); }

void CommandListImpl::drawIndexed(uint32_t indexCount) { vkCmdDrawIndexed(cb, indexCount, 1, 0, 0, 0); }

void CommandListImpl::copyImageToBuffer(Image& imageBase, Buffer& bufferBase) {
    auto& image = static_cast<ImageImpl&>(imageBase);
    auto& buffer = static_cast<BufferImpl&>(bufferBase);
    if (image.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        throw Error("copy source '" + image.name() + "' is not in TransferSrc state");
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {image.width(), image.height(), 1};
    vkCmdCopyImageToBuffer(cb, image.image, image.layout, buffer.buffer, 1, &region);
}

void CommandListImpl::copyBufferToImage(Buffer& bufferBase, Image& imageBase) {
    auto& buffer = static_cast<BufferImpl&>(bufferBase);
    auto& image = static_cast<ImageImpl&>(imageBase);
    if (image.layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        throw Error("upload target '" + image.name() + "' is not in TransferDst state");
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(image.mipLevels);
    VkDeviceSize offset = 0;
    for (uint32_t level = 0; level < image.mipLevels; level++) {
        VkBufferImageCopy r{};
        r.bufferOffset = offset;
        r.imageSubresource = {image.aspect, level, 0, 1};
        r.imageExtent = {std::max(1u, image.width() >> level), std::max(1u, image.height() >> level), 1};
        regions.push_back(r);
        offset += static_cast<VkDeviceSize>(r.imageExtent.width) * r.imageExtent.height * 4;
    }
    if (offset > buffer.size()) throw Error("upload buffer too small for image '" + image.name() + "'");
    vkCmdCopyBufferToImage(cb, buffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           static_cast<uint32_t>(regions.size()), regions.data());
}

}  // namespace vk

// Factory for the one backend that exists so far (declared in rhi.h, defined in vk_device.cpp).

}  // namespace rhi
}  // namespace engine