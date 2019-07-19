#pragma once

#ifndef MEPHNDEBUG
#define SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>

namespace mephisto {
  template<typename... Args>
  void log(Args&&... args) {
    spdlog::get("alpaka")->info(std::forward<Args>(args)...);
  }

  void initlog() {
    if(spdlog::get("alpaka") == nullptr) {
      spdlog::set_pattern("[%S.%e][%t] %v");
      spdlog::stdout_logger_mt("alpaka");
    }
  }
}
#else
namespace mephisto {
  template<typename... Args>
  void log(Args&&... args) { }
  void initlog() { }
}

#endif

