#include "rg/render_graph.h"

#include <sstream>

namespace engine {
namespace rg {

PassBuilder& PassBuilder::read(Handle image) {
    detail::Access a;
    a.image = image;
    a.write = false;
    a.state = rhi::ImageState::ShaderRead;
    pass_.accesses.push_back(a);
    return *this;
}

PassBuilder& PassBuilder::colorTarget(Handle image, rhi::LoadOp load, const std::array<float, 4>& clear) {
    detail::Access a;
    a.image = image;
    a.write = true;
    a.state = rhi::ImageState::ColorTarget;
    a.load = load;
    a.clear = clear;
    pass_.accesses.push_back(a);
    return *this;
}

PassBuilder& PassBuilder::depthTarget(Handle image) {
    detail::Access a;
    a.image = image;
    a.write = true;
    a.state = rhi::ImageState::DepthTarget;
    a.load = rhi::LoadOp::Clear;
    pass_.accesses.push_back(a);
    return *this;
}

PassBuilder& PassBuilder::runOnce() {
    pass_.runOnce = true;
    return *this;
}

PassBuilder& PassBuilder::execute(std::function<void(rhi::CommandList&)> body) {
    pass_.body = std::move(body);
    return *this;
}

Handle Graph::import(rhi::Image& image, const std::string& name) {
    Slot s;
    s.image = &image;
    s.name = name;
    slots_.push_back(std::move(s));
    compiled_ = false;
    return static_cast<Handle>(slots_.size() - 1);
}

Handle Graph::create(rhi::Device& device, const rhi::ImageDesc& desc) {
    Slot s;
    s.owned = device.createImage(desc);
    s.image = s.owned.get();
    s.name = desc.name;
    slots_.push_back(std::move(s));
    compiled_ = false;
    return static_cast<Handle>(slots_.size() - 1);
}

PassBuilder Graph::addPass(const std::string& name) {
    passes_.push_back(std::make_unique<detail::Pass>());
    passes_.back()->name = name;
    compiled_ = false;
    return PassBuilder(*passes_.back());
}

rhi::Image& Graph::image(Handle h) {
    if (h >= slots_.size()) throw rhi::Error("bad image handle");
    return *slots_[h].image;
}

void Graph::compile() {
    // An image is available once it is imported or written by an earlier pass.
    std::vector<bool> available(slots_.size(), false);
    for (size_t i = 0; i < slots_.size(); i++)
        if (slots_[i].owned == nullptr) available[i] = true;

    for (const auto& pass : passes_) {
        if (!pass->body) throw rhi::Error("pass '" + pass->name + "' has no body");
        for (const auto& a : pass->accesses) {
            if (a.image >= slots_.size()) throw rhi::Error("pass '" + pass->name + "' uses a bad handle");
            if (!a.write && !available[a.image])
                throw rhi::Error("pass '" + pass->name + "' reads '" + slots_[a.image].name +
                                 "' before any earlier pass writes it");
            for (const auto& b : pass->accesses)
                if (&a != &b && a.image == b.image && (a.write || b.write))
                    throw rhi::Error("pass '" + pass->name + "' both reads and writes '" + slots_[a.image].name + "'");
        }
        for (const auto& a : pass->accesses)
            if (a.write) available[a.image] = true;
    }
    compiled_ = true;
}

void Graph::execute(rhi::CommandList& cmd) {
    if (!compiled_) compile();
    // With profiling, each pass is bracketed by two timestamps (2 per pass, in declaration order). A pass that is
    // skipped (runOnce, already run) records its two timestamps back to back, so its time is zero.
    if (profile_) cmd.resetTimestamps(static_cast<uint32_t>(2 * passes_.size()));
    uint32_t index = 0;
    for (const auto& pass : passes_) {
        const uint32_t first = 2 * index++;
        if (profile_) cmd.writeTimestamp(first);
        if (!(pass->runOnce && pass->ran)) {
// Move every input and output into the state this pass needs.
            const detail::Access* colour = nullptr;
            const detail::Access* depth = nullptr;
            for (const auto& a : pass->accesses) {
                cmd.transition(*slots_[a.image].image, a.state);
                if (a.state == rhi::ImageState::ColorTarget) colour = &a;
                if (a.state == rhi::ImageState::DepthTarget) depth = &a;
            }

            if (colour) {
                rhi::Image* depthImage = depth ? slots_[depth->image].image : nullptr;
                cmd.beginRendering(*slots_[colour->image].image, colour->load, colour->clear, depthImage);
                pass->body(cmd);
                cmd.endRendering();
            } else {
                pass->body(cmd);
            }
            pass->ran = true;
        }
        if (profile_) cmd.writeTimestamp(first + 1);
    }
}

std::vector<std::string> Graph::passNames() const {
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto& pass : passes_) names.push_back(pass->name);
    return names;
}

std::string Graph::describe() const {
    std::ostringstream os;
    int index = 0;
    for (const auto& pass : passes_) {
        os << index++ << ". " << pass->name << (pass->runOnce ? " [once]" : "") << "\n";
        for (const auto& a : pass->accesses) {
            const char* what = a.state == rhi::ImageState::ColorTarget   ? "colour"
                               : a.state == rhi::ImageState::DepthTarget ? "depth"
                                                                         : "read";
            os << "     " << what << " " << slots_[a.image].name << "\n";
        }
    }
    return os.str();
}

}  // namespace rg
}  // namespace engine
