// Render graph: passes declare the images they read and write, in the order they run.
// The graph checks the declarations once, then each frame it moves every image into the
// state its pass needs (barriers come from the RHI), opens the rendering scope, and runs
// the pass body. The graph owns the images it creates, but does not alias memory: each
// image keeps its own storage for the whole frame. Aliasing is a later step.
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rhi/rhi.h"

namespace engine {
namespace rg {

using Handle = uint32_t;

namespace detail {

struct Access {
    Handle image = 0;
    bool write = false;  // true: colour or depth target; false: sampled
    rhi::ImageState state = rhi::ImageState::ShaderRead;
    rhi::LoadOp load = rhi::LoadOp::Clear;
    std::array<float, 4> clear = {0, 0, 0, 0};
};

struct Pass {
    std::string name;
    std::vector<Access> accesses;
    bool runOnce = false;
    bool ran = false;
    std::function<void(rhi::CommandList&)> body;
};

}  // namespace detail

class PassBuilder {
public:
    // The pass samples the image in a shader.
    PassBuilder& read(Handle image);
    // The pass renders into the image. Clear fills it first; Load keeps existing contents.
    PassBuilder& colorTarget(Handle image, rhi::LoadOp load, const std::array<float, 4>& clear = {0, 0, 0, 0});
    // The pass renders depth into the image. The depth buffer is cleared to 1 at the start.
    PassBuilder& depthTarget(Handle image);
    // The body runs on the first execute() only (for tables that do not change).
    PassBuilder& runOnce();
    // The pass body. Called inside the rendering scope (when a colour target is declared).
    PassBuilder& execute(std::function<void(rhi::CommandList&)> body);

private:
    friend class Graph;
    explicit PassBuilder(detail::Pass& p) : pass_(p) {}
    detail::Pass& pass_;
};

class Graph {
public:
    // Adds an image owned by the caller. It must outlive the graph.
    Handle import(rhi::Image& image, const std::string& name);
    // Creates an image owned by the graph.
    Handle create(rhi::Device& device, const rhi::ImageDesc& desc);

    PassBuilder addPass(const std::string& name);

    // Checks the declarations: every read has an earlier writer or an import, and no pass
    // both reads and writes one image. Throws rhi::Error. Called by execute() on first use.
    void compile();
    // Records every pass into cmd in declaration order, with the needed barriers.
    void execute(rhi::CommandList& cmd);
    // With profiling on, execute() brackets every pass with two GPU timestamps (2 per pass, in declaration order).
    void setProfiling(bool on) { profile_ = on; }
    std::vector<std::string> passNames() const;

    rhi::Image& image(Handle h);
    // Text dump of the passes and their accesses.
    std::string describe() const;

private:
    struct Slot {
        rhi::Image* image = nullptr;
        std::unique_ptr<rhi::Image> owned;
        std::string name;
    };

    std::vector<Slot> slots_;
    // Pass storage is a vector of unique_ptr so PassBuilder references stay valid.
    std::vector<std::unique_ptr<detail::Pass>> passes_;
    bool compiled_ = false;
    bool profile_ = false;
};

}  // namespace rg
}  // namespace engine
