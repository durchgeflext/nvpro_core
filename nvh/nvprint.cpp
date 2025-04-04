/*
 * Copyright (c) 2014-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "nvprint.hpp"
#include "fileoperations.hpp"

#include <limits.h>
#include <mutex>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <signal.h>
#include <unistd.h>
#endif

enum class TriState
{
  eUnknown,
  eFalse,
  eTrue
};

static std::string       s_logFileName;
static std::vector<char> s_strBuffer;  // Persistent allocation for formatted text.
#ifdef _WIN32
static std::vector<wchar_t> s_wideStrBuffer;  // Persistent allocation for UTF-16 text.
#endif
static FILE*               s_fd                   = nullptr;
static bool                s_bLogReady            = false;
static bool                s_bPrintLogging        = true;
static uint32_t            s_bPrintFileLogging    = LOGBITS_ALL;
static uint32_t            s_bPrintConsoleLogging = LOGBITS_ALL;
static uint32_t            s_bPrintBreakpoints    = 0;
static int                 s_printLevel           = -1;  // <0 mean no level prefix
static PFN_NVPRINTCALLBACK s_printCallback        = nullptr;
static TriState            s_consoleSupportsColor = TriState::eUnknown;
// Lock this when modifying any static variables.
// Because it is a recursive mutex, its owner can lock it multiple times.
static std::recursive_mutex s_mutex;

void nvprintSetLogFileName(const char* name) noexcept
{
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);

  if(name == NULL || s_logFileName == name)
    return;

  try
  {
    s_logFileName = name;
  }
  catch(const std::exception& e)
  {
    nvprintLevel(LOGLEVEL_ERROR, "nvprintfSetLogFileName could not allocate space for new file name. Additional info below:");
    nvprintLevel(LOGLEVEL_ERROR, e.what());
  }

  if(s_fd)
  {
    fclose(s_fd);
    s_fd        = nullptr;
    s_bLogReady = false;
  }
}
void nvprintSetCallback(PFN_NVPRINTCALLBACK callback)
{
  s_printCallback = callback;
}
void nvprintSetLevel(int l)
{
  s_printLevel = l;
}
int nvprintGetLevel()
{
  return s_printLevel;
}
void nvprintSetLogging(bool b)
{
  s_bPrintLogging = b;
}

void nvprintSetFileLogging(bool state, uint32_t mask)
{
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);

  if(state)
  {
    s_bPrintFileLogging |= mask;
  }
  else
  {
    s_bPrintFileLogging &= ~mask;
  }
}

void nvprintSetConsoleLogging(bool state, uint32_t mask)
{
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);

  if(state)
  {
    s_bPrintConsoleLogging |= mask;
  }
  else
  {
    s_bPrintConsoleLogging &= ~mask;
  }
}

void nvprintSetBreakpoints(bool state, uint32_t mask)
{
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);

  if(state)
  {
    s_bPrintBreakpoints |= mask;
  }
  else
  {
    s_bPrintBreakpoints &= ~mask;
  }
}

void nvprintfV(va_list& vlist, const char* fmt, int level) noexcept
{
  if(s_bPrintLogging == false)
  {
    return;
  }

  // Format the inputs into s_strBuffer.
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);
  {
    // Copy vlist as it may be modified by vsnprintf.
    va_list vlistCopy;
    va_copy(vlistCopy, vlist);
    const int charactersNeeded = vsnprintf(s_strBuffer.data(), s_strBuffer.size(), fmt, vlistCopy);
    va_end(vlistCopy);

    // Check that:
    // * vsnprintf did not return an error;
    // * The string (plus null terminator) could fit in a vector.
    if((charactersNeeded < 0) || (size_t(charactersNeeded) > s_strBuffer.max_size() - 1))
    {
      // Formatting error
      nvprintLevel(LOGLEVEL_ERROR, "nvprintfV: Internal message formatting error.");
      return;
    }

    // Increase the size of s_strBuffer as needed if there wasn't enough space.
    if(size_t(charactersNeeded) >= s_strBuffer.size())
    {
      try
      {
        // Make sure to add 1, because vsnprintf doesn't count the terminating
        // null character. This can potentially throw an exception.
        s_strBuffer.resize(size_t(charactersNeeded) + 1, '\0');
      }
      catch(const std::exception& e)
      {
        nvprintLevel(LOGLEVEL_ERROR, "nvprintfV: Error resizing buffer to hold message. Additional info below:");
        nvprintLevel(LOGLEVEL_ERROR, e.what());
        return;
      }

      // Now format it; we know this will succeed.
      (void)vsnprintf(s_strBuffer.data(), s_strBuffer.size(), fmt, vlist);
    }
  }

  nvprintLevel(level, s_strBuffer.data());
}

void nvprintLevel(int level, const std::string& msg) noexcept
{
  nvprintLevel(level, msg.c_str());
}

#ifdef _WIN32
static void printDebugString(const char* utf8_str) noexcept
{
  // Convert our text from UTF-8 to UTF-16, so we can pass it to
  // OutputDebugStringW.
  // We call Windows' functions here directly instead of using <codecvt>,
  // because <codecvt> is deprecated and scheduled to be removed from C++.
  const size_t utf8_bytes = strlen(utf8_str);
  if(utf8_bytes == 0)
  {
    return;
  }
  if(utf8_bytes > INT_MAX)
  {
    OutputDebugStringW(L"Could not format text as UTF-16: input was longer than INT_MAX bytes.\n");
    return;
  }
  const int utf8_bytes_int   = static_cast<int>(utf8_bytes);
  const int utf16_characters = MultiByteToWideChar(CP_UTF8, 0, utf8_str, utf8_bytes_int, nullptr, 0);
  if(utf16_characters <= 0)
  {
    OutputDebugStringW(L"Could not format text as UTF-16: MultiByteToWideChar() returned an error code or a negative number of bytes.\n");
    return;
  }
  const size_t utf16_characters_sizet = static_cast<size_t>(utf16_characters);

  // Make sure s_wideStrBuffer contains space for utf16_characters_sizet plus
  // a terminating null character.
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);
  if(utf16_characters_sizet >= s_wideStrBuffer.size())
  {
    try
    {
      s_wideStrBuffer.resize(utf16_characters_sizet + 1, L'\0');
    }
    catch(const std::exception& /* unused */)
    {
      OutputDebugStringW(L"Could not format text as UTF-16: Out of memory exception when allocating UTF-16 buffer.\n");
      return;
    }
  }
  std::ignore = MultiByteToWideChar(CP_UTF8, 0, utf8_str, utf8_bytes_int, s_wideStrBuffer.data(), utf16_characters);
  // Write the terminating null character:
  s_wideStrBuffer[utf16_characters] = L'\0';
  OutputDebugStringW(s_wideStrBuffer.data());
}
#endif

void nvprintLevel(int level, const char* msg) noexcept
{
  std::lock_guard<std::recursive_mutex> lockGuard(s_mutex);

#ifdef WIN32
  printDebugString(msg);
#endif

  if(s_bPrintFileLogging & (1 << level))
  {
    if(s_bLogReady == false)
    {

      // Set a default log file name if none was set.
      if(s_logFileName.empty())
      {
        try
        {
          std::filesystem::path       exePath = nvh::getExecutablePath();
          const std::filesystem::path pathLog = exePath.parent_path() / ("log_" + exePath.stem().string() + ".txt");
          s_logFileName                       = pathLog.string();
        }
        catch(const std::exception& e)
        {
#ifdef WIN32
          OutputDebugStringW(L"Could not allocate space for the default log file name.\n");
          printDebugString(e.what());
          // maybe even increase level to LOGLEVEL_ERROR?
#endif
          return;
        }
      }


      s_fd        = fopen(s_logFileName.c_str(), "wt");
      s_bLogReady = true;
    }
    if(s_fd)
    {
      fputs(msg, s_fd);
    }
  }

  if(s_printCallback)
  {
    s_printCallback(level, msg);
  }

  if(s_bPrintConsoleLogging & (1 << level))
  {
    // Determine if the output supports ANSI color sequences only once to avoid
    // many calls to isatty.
    if(TriState::eUnknown == s_consoleSupportsColor)
    {
      // Determining this perfectly is difficult; terminfo does it by storing
      // a large table of all consoles it knows about. For now, we assume
      // all consoles support colors, and all pipes do not.
#ifdef WIN32
      bool supportsColor = _isatty(_fileno(stderr)) && _isatty(_fileno(stdout));
      // This enables ANSI escape codes from the app side.
      // We do this because on Windows 10, cmd.exe is a console, but only
      // supports ANSI escape codes by default if the
      // HKEY_CURRENT_USER\Console\VirtualTerminalLevel registry key is
      // nonzero, which we don't want to assume.
      // See https://github.com/nvpro-samples/vk_raytrace/issues/28.
      // On failure, turn off colors.
      if(supportsColor)
      {
        for(DWORD stdHandleIndex : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE})
        {
          const HANDLE consoleHandle = GetStdHandle(stdHandleIndex);
          if(INVALID_HANDLE_VALUE == consoleHandle)
          {
            supportsColor = false;
            break;
          }
          DWORD consoleMode = 0;
          if(0 == GetConsoleMode(consoleHandle, &consoleMode))
          {
            supportsColor = false;
            break;
          }
          SetConsoleMode(consoleHandle, consoleMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
      }
#else
      const bool supportsColor = isatty(fileno(stderr)) && isatty(fileno(stdout));
#endif
      s_consoleSupportsColor = (supportsColor ? TriState::eTrue : TriState::eFalse);
    }

    FILE* outStream = (((1 << level) & LOGBITS_ERRORS) ? stderr : stdout);

    if(TriState::eTrue == s_consoleSupportsColor)
    {
      // Set the foreground color depending on level:
      // https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters
      if(level == LOGLEVEL_OK)
      {
        fputs("\033[32m", outStream);  // Green
      }
      else if(level == LOGLEVEL_ERROR)
      {
        fputs("\033[31m", outStream);  // Red
      }
      else if(level == LOGLEVEL_WARNING)
      {
        fputs("\033[33m", outStream);  // Yellow
      }
      else if(level == LOGLEVEL_DEBUG)
      {
        fputs("\033[36m", outStream);  // Cyan
      }
    }

    fputs(msg, outStream);

    if(TriState::eTrue == s_consoleSupportsColor)
    {
      // Reset all attributes
      fputs("\033[0m", outStream);
    }
  }

  if(s_bPrintBreakpoints & (1 << level))
  {
#ifdef WIN32
    DebugBreak();
#else
    raise(SIGTRAP);
#endif
  }
}

void nvprintf(
#ifdef _MSC_VER
    _Printf_format_string_
#endif
    const char* fmt,
    ...) noexcept
{
  //    int r = 0;
  va_list vlist;
  va_start(vlist, fmt);
  nvprintfV(vlist, fmt, s_printLevel);
  va_end(vlist);
}
void nvprintfLevel(int level,
#ifdef _MSC_VER
                   _Printf_format_string_
#endif
                   const char* fmt,
                   ...) noexcept
{
  va_list vlist;
  va_start(vlist, fmt);
  nvprintfV(vlist, fmt, level);
  va_end(vlist);
}
