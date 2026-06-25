#ifndef SUNMMIO_PIPELINE_UTILS_H
#define SUNMMIO_PIPELINE_UTILS_H

#include <string>

namespace tvm {
namespace tl {
inline int name2iter(const std::string &name) {
  return std::stoi(name.substr(0, name.find('-')));
}

inline int name2id(const std::string &name) {
  return std::stoi(name.substr(name.find('-') + 1));
}

} // namespace tl
} // namespace tvm
#endif
