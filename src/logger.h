#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

inline void init_logger() {
    // Create a color multi-threaded stdout sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);

    // Create a basic multi-threaded file sink
    auto basic_file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("app.log", true);
    basic_file_sink->set_level(spdlog::level::trace);

    // Combine the sinks into a vector
    std::vector<spdlog::sink_ptr> sinks{console_sink, basic_file_sink};

    // Create a logger with the combined sinks
    auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::trace); // Set the overall logger level (must be lowest of all sinks)

    // Optional: Register the logger globally so it can be accessed by name
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    spdlog::flush_on(spdlog::level::info);
}
