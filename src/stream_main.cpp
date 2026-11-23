// Live transcription with the streaming API.
//
// Microphone (raw 16 kHz mono s16le on stdin):
//   arecord -q -f S16_LE -r 16000 -c 1 -t raw | ./build/voxtral-stream --model M
// A WAV file, fed at real-time pace (to measure latency):
//   ./build/voxtral-stream --model M --audio clip.wav [--timestamps]

// Server mode (one process, many utterances; the model stays loaded):
//   ./build/voxtral-stream --model M --server
// stdin carries frames: 1 byte type + 4 bytes little-endian length + payload.
//   'B' begin an utterance, 'A' audio (16 kHz mono s16le), 'E' end it (flush),
//   'C' cancel it, 'Q' quit.
// stdout gets one JSON object per line: {"ready":true} once the model is
// loaded, {"partial":"..."} with the transcript so far whenever it grows,
// {"final":"..."} after 'E', and {"error":"..."}.

#include "voxtral.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;

namespace {

struct options {
    std::string         model;
    std::string         audio;
    voxtral_gpu_backend gpu        = voxtral_gpu_backend::auto_detect;
    int32_t             step_ms    = 240;
    bool                timestamps = false;
    bool                server     = false;
};

bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto value = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--model") {
            const char * v = value();
            if (!v) return false;
            o.model = v;
        } else if (a == "--audio") {
            const char * v = value();
            if (!v) return false;
            o.audio = v;
        } else if (a == "--step-ms") {
            const char * v = value();
            if (!v) return false;
            o.step_ms = std::max(80, std::atoi(v));
        } else if (a == "--gpu") {
            const char * v = value();
            if (!v) return false;
            const std::string g = v;
            if (g == "auto") o.gpu = voxtral_gpu_backend::auto_detect;
            else if (g == "vulkan") o.gpu = voxtral_gpu_backend::vulkan;
            else if (g == "cuda") o.gpu = voxtral_gpu_backend::cuda;
            else if (g == "metal") o.gpu = voxtral_gpu_backend::metal;
            else if (g == "none") o.gpu = voxtral_gpu_backend::none;
            else return false;
        } else if (a == "--timestamps") {
            o.timestamps = true;
        } else if (a == "--server") {
            o.server = true;
        } else {
            return false;
        }
    }
    return !o.model.empty();
}

// Length of the longest prefix of s that ends on a complete UTF-8 character.
size_t complete_utf8_prefix(const std::string & s) {
    size_t i = s.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char) s[i - 1];
        if ((c & 0xC0) != 0x80) {
            const size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
            return back + 1 >= need ? s.size() : i - 1;
        }
        i--;
        back++;
    }
    return s.size();
}

bool read_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> data;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    // Find the "data" chunk; assumes 16-bit PCM, 16 kHz, mono.
    for (size_t i = 12; i + 8 <= data.size();) {
        const uint32_t size = data[i + 4] | (data[i + 5] << 8) | (data[i + 6] << 16) | ((uint32_t) data[i + 7] << 24);
        if (std::memcmp(&data[i], "data", 4) == 0) {
            const size_t end = std::min(data.size(), i + 8 + (size_t) size);
            for (size_t j = i + 8; j + 1 < end; j += 2) {
                out.push_back((int16_t) (data[j] | (data[j + 1] << 8)) / 32768.0f);
            }
            return true;
        }
        i += 8 + size + (size & 1);
    }
    return false;
}


std::string json_string(const std::string & text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}

void send_line(const std::string & key, const std::string & value) {
    std::printf("{\"%s\":%s}\n", key.c_str(), value.c_str());
    std::fflush(stdout);
}

bool read_exact(void * dst, size_t n) {
    return std::fread(dst, 1, n, stdin) == n;
}

int run_server(voxtral_context * ctx, int32_t step_samples) {
    struct event {
        char               type;
        std::vector<float> audio;
    };
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<event> events;
    bool closed = false;

    std::thread reader([&]() {
        std::vector<uint8_t> payload;
        for (;;) {
            uint8_t head[5];
            if (!read_exact(head, sizeof(head))) break;
            const uint32_t len = head[1] | (head[2] << 8) | (head[3] << 16) | ((uint32_t) head[4] << 24);
            payload.resize(len);
            if (len > 0 && !read_exact(payload.data(), len)) break;
            event e{(char) head[0], {}};
            if (e.type == 'A') {
                e.audio.resize(len / 2);
                for (size_t i = 0; i < e.audio.size(); i++) {
                    e.audio[i] = (int16_t) (payload[2 * i] | (payload[2 * i + 1] << 8)) / 32768.0f;
                }
            }
            std::lock_guard<std::mutex> lock(mtx);
            events.push_back(std::move(e));
            cv.notify_one();
            if (head[0] == 'Q') break;
        }
        std::lock_guard<std::mutex> lock(mtx);
        closed = true;
        cv.notify_one();
    });

    send_line("ready", "true");

    voxtral_stream * s = nullptr;
    std::string sent;
    int32_t unstepped = 0;
    bool quit = false;
    while (!quit) {
        std::deque<event> batch;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return closed || !events.empty(); });
            if (events.empty() && closed) break;
            batch.swap(events);
        }
        // Audio that piled up during a slow step goes into one step, so the
        // transcript never falls behind the speech.
        bool want_step = false;
        for (event & e : batch) {
            switch (e.type) {
                case 'B':
                    voxtral_stream_free(s);
                    s = voxtral_stream_init(ctx);
                    sent.clear();
                    unstepped = 0;
                    if (!s) send_line("error", json_string("could not start a stream"));
                    break;
                case 'A':
                    if (s) {
                        voxtral_stream_feed(s, e.audio.data(), (int32_t) e.audio.size());
                        unstepped += (int32_t) e.audio.size();
                        want_step = want_step || unstepped >= step_samples;
                    }
                    break;
                case 'E': {
                    if (!s) {
                        send_line("final", json_string(""));
                        break;
                    }
                    std::string text;
                    if (!voxtral_stream_step(s, true, text)) {
                        send_line("error", json_string("transcription failed"));
                        text.clear();
                    }
                    send_line("final", json_string(text));
                    voxtral_stream_free(s);
                    s = nullptr;
                    want_step = false;
                    break;
                }
                case 'C':
                    voxtral_stream_free(s);
                    s = nullptr;
                    want_step = false;
                    break;
                case 'Q':
                    quit = true;
                    break;
                default:
                    send_line("error", json_string(std::string("unknown frame ") + e.type));
            }
        }
        if (want_step && s) {
            std::string text;
            unstepped = 0;
            if (!voxtral_stream_step(s, false, text)) {
                send_line("error", json_string("transcription failed"));
                continue;
            }
            const size_t upto = complete_utf8_prefix(text);
            if (upto > sent.size()) {
                sent = text.substr(0, upto);
                send_line("partial", json_string(sent));
            }
        }
    }
    voxtral_stream_free(s);
    // The reader may still be blocked on stdin; the process is exiting anyway.
    reader.detach();
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) {
        std::cerr << "usage: voxtral-stream --model M [--audio file.wav] [--gpu auto|vulkan|cuda|metal|none]\n"
                     "                      [--step-ms 240] [--timestamps] [--server]\n"
                     "without --audio, reads raw 16 kHz mono s16le from stdin\n";
        return 1;
    }

    // By default ggml-vulkan places buffers in host-visible VRAM (ReBAR). On
    // AMD/RADV the driver may keep part of it in system memory, and a growing
    // encoder graph then runs 5-10x slower past ~13 s of audio. Device-only
    // memory keeps every step on the GPU.
    setenv("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1", 0);

    voxtral_log_callback logger = [](voxtral_log_level level, const std::string & msg) {
        if (level <= voxtral_log_level::warn) std::cerr << "voxtral: " << msg << "\n";
    };
    voxtral_model * model = voxtral_model_load_from_file(o.model, logger, o.gpu);
    if (!model) return 2;
    voxtral_context_params cp;
    cp.log_level = voxtral_log_level::warn;
    cp.logger = logger;
    cp.gpu = o.gpu;
    voxtral_context * ctx = voxtral_init_from_model(model, cp);
    if (!ctx) return 3;
    const int32_t step_samples = VOXTRAL_SAMPLE_RATE * o.step_ms / 1000;
    if (o.server) {
        const int code = run_server(ctx, step_samples);
        voxtral_free(ctx);
        voxtral_model_free(model);
        return code;
    }
    voxtral_stream * s = voxtral_stream_init(ctx);
    if (!s) return 4;

    std::vector<float> file_audio;
    if (!o.audio.empty() && !read_wav_16k_mono(o.audio, file_audio)) {
        std::cerr << "could not read " << o.audio << " (expected 16-bit PCM WAV)\n";
        return 5;
    }

    if (o.audio.empty()) std::fprintf(stderr, "Pode falar (Ctrl+C para terminar).\n\n");
    size_t printed = 0;
    size_t fed = 0;
    std::vector<double> step_times;
    const auto t0 = clock_type::now();
    auto seconds = [&]() { return std::chrono::duration<double>(clock_type::now() - t0).count(); };

    auto emit = [&](const std::string & text, bool final) {
        const size_t upto = final ? text.size() : complete_utf8_prefix(text);
        if (upto <= printed) return;
        const std::string delta = text.substr(printed, upto - printed);
        if (o.timestamps) {
            std::printf("[%5.2fs falado %5.2fs] %s\n", seconds(), (double) fed / VOXTRAL_SAMPLE_RATE, delta.c_str());
        } else {
            std::fputs(delta.c_str(), stdout);
        }
        std::fflush(stdout);
        printed = upto;
    };

    auto step = [&](bool final) {
        std::string text;
        const auto ts = clock_type::now();
        if (!voxtral_stream_step(s, final, text)) return false;
        step_times.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - ts).count());
        emit(text, final);
        return true;
    };

    bool ok = true;
    if (!file_audio.empty()) {
        // Real-time pace, like the microphone: each step takes all the audio
        // that would have been spoken by now (at least step_ms of it).
        while (ok && fed < file_audio.size()) {
            const size_t min_end = std::min(file_audio.size(), fed + (size_t) step_samples);
            const double wait = (double) min_end / VOXTRAL_SAMPLE_RATE - seconds();
            if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
            const size_t spoken = (size_t) (seconds() * VOXTRAL_SAMPLE_RATE);
            const size_t end = std::min(file_audio.size(), std::max(min_end, spoken));
            voxtral_stream_feed(s, file_audio.data() + fed, (int32_t) (end - fed));
            fed = end;
            ok = step(false);
        }
    } else {
        // The microphone is read on its own thread: when a step takes longer
        // than step_ms, the next one takes everything that arrived meanwhile
        // instead of falling behind.
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<float> pending;
        bool eof = false;
        std::thread reader([&]() {
            std::vector<int16_t> raw(VOXTRAL_SAMPLE_RATE / 50);
            size_t n;
            while ((n = std::fread(raw.data(), sizeof(int16_t), raw.size(), stdin)) > 0) {
                std::lock_guard<std::mutex> lock(mtx);
                for (size_t i = 0; i < n; i++) pending.push_back(raw[i] / 32768.0f);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lock(mtx);
            eof = true;
            cv.notify_one();
        });
        std::vector<float> chunk;
        while (ok) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]() { return eof || (int32_t) pending.size() >= step_samples; });
                chunk.swap(pending);
                pending.clear();
            }
            if (!chunk.empty()) {
                voxtral_stream_feed(s, chunk.data(), (int32_t) chunk.size());
                fed += chunk.size();
                ok = step(false);
            }
            std::lock_guard<std::mutex> lock(mtx);
            if (eof && pending.empty()) break;
        }
        reader.join();
    }
    const double before_final = seconds();
    ok = ok && step(true);
    std::printf("\n");
    std::fflush(stdout);

    if (!step_times.empty()) {
        double sum = 0, worst = 0;
        for (double t : step_times) { sum += t; worst = std::max(worst, t); }
        std::fprintf(stderr, "[stream] %.1fs de áudio, %zu passos, média %.0f ms, pior %.0f ms, final %.0f ms\n",
            (double) fed / VOXTRAL_SAMPLE_RATE, step_times.size(), sum / step_times.size(), worst,
            (seconds() - before_final) * 1000);
    }

    voxtral_stream_free(s);
    voxtral_free(ctx);
    voxtral_model_free(model);
    return ok ? 0 : 6;
}
