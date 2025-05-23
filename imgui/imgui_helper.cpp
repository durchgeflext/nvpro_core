/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#define GLFW_INCLUDE_NONE
#include "imgui/imgui_helper.h"
#include <backends/imgui_impl_glfw.h>
#include <GLFW/glfw3.h>
#include <math.h>


#include <fstream>

namespace ImGuiH {

void Init(int width, int height, void* userData, FontMode fontmode)
{
  ImGui::CreateContext();
  setFonts(fontmode);
  auto& imgui_io       = ImGui::GetIO();
  imgui_io.IniFilename = nullptr;
  imgui_io.UserData    = userData;
  imgui_io.DisplaySize = ImVec2(float(width), float(height));
  imgui_io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable keyboard controls (tab, space, arrow keys)

  // Scale style sizes for high-DPI monitors
  ImGuiStyle& imgui_style = ImGui::GetStyle();
  imgui_style.ScaleAllSizes(getDPIScale());
}

void Deinit()
{
  ImGui::DestroyContext(nullptr);
}


bool Combo(const char* label, size_t numEnums, const Enum* enums, void* valuePtr, ImGuiComboFlags flags, ValueType valueType, bool* valueChanged)
{
  int*   ivalue = (int*)valuePtr;
  float* fvalue = (float*)valuePtr;

  size_t idx     = 0;
  bool   found   = false;
  bool   changed = false;
  for(size_t i = 0; i < numEnums; i++)
  {
    switch(valueType)
    {
      case TYPE_INT:
        if(enums[i].ivalue == *ivalue)
        {
          idx   = i;
          found = true;
        }
        break;
      case TYPE_FLOAT:
        if(enums[i].fvalue == *fvalue)
        {
          idx   = i;
          found = true;
        }
        break;
      default:
        break;
    }
  }

  if(!found)
  {
    assert(!"No such value in combo!");
    return false;
  }

  if(ImGui::BeginCombo(label, enums[idx].name.c_str(), flags))  // The second parameter is the label previewed before opening the combo.
  {
    for(size_t i = 0; i < numEnums; i++)
    {
      ImGui::BeginDisabled(enums[i].disabled);
      bool is_selected = i == idx;
      if(ImGui::Selectable(enums[i].name.c_str(), is_selected))
      {
        switch(valueType)
        {
          case TYPE_INT:
            *ivalue = enums[i].ivalue;
            break;
          case TYPE_FLOAT:
            *fvalue = enums[i].fvalue;
            break;
        }

        changed = true;
      }
      if(is_selected)
      {
        ImGui::SetItemDefaultFocus();  // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
      }
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }

  if(valueChanged)
    *valueChanged = changed;

  return changed;
}

//--------------------------------------------------------------------------------------------------
//
// If GLFW has been initialized, returns the DPI scale of the primary monitor. Otherwise, returns 1.
//
float getDPIScale()
{
  // Cached DPI scale, so that this doesn't change after the first time code calls getDPIScale.
  // A negative value indicates that the value hasn't been computed yet.
  static float cached_dpi_scale = -1.0f;

  if(cached_dpi_scale < 0.0f)
  {
    // Compute the product of the monitor DPI scale and any DPI scale
    // set in the NVPRO_DPI_SCALE variable.
    cached_dpi_scale = 1.0f;

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    assert(monitor);
    if(monitor != nullptr)
    {
      float y_scale;
      glfwGetMonitorContentScale(monitor, &cached_dpi_scale, &y_scale);
    }
    // Otherwise, GLFW isn't initialized yet, but might be in the future.
    // (Note that this code assumes all samples use GLFW.)

    // Multiply by the value of the NVPRO_DPI_SCALE environment variable.
    const char* dpi_env = getenv("NVPRO_DPI_SCALE");
    if(dpi_env)
    {
      const float parsed_dpi_env = strtof(dpi_env, nullptr);
      if(parsed_dpi_env != 0.0f)
      {
        cached_dpi_scale *= parsed_dpi_env;
      }
    }

    cached_dpi_scale = (cached_dpi_scale > 0.0f ? cached_dpi_scale : 1.0f);
  }

  return cached_dpi_scale;
}

//--------------------------------------------------------------------------------------------------
// Setting a dark style for the GUI
// The colors were coded in sRGB color space, set the useLinearColor
// flag to convert to linear color space.
void setStyle(bool useLinearColor)
{
  typedef ImVec4 (*srgbFunction)(float, float, float, float);
  srgbFunction   passthrough = [](float r, float g, float b, float a) -> ImVec4 { return ImVec4(r, g, b, a); };
  srgbFunction   toLinear    = [](float r, float g, float b, float a) -> ImVec4 {
    auto toLinearScalar = [](float u) -> float {
      return u <= 0.04045 ? 25 * u / 323.f : powf((200 * u + 11) / 211.f, 2.4f);
    };
    return ImVec4(toLinearScalar(r), toLinearScalar(g), toLinearScalar(b), a);
  };
  srgbFunction srgb = useLinearColor ? toLinear : passthrough;

  ImGui::StyleColorsDark();

  ImGuiStyle& style                  = ImGui::GetStyle();
  style.WindowRounding               = 0.0f;
  style.WindowBorderSize             = 0.0f;
  style.ColorButtonPosition          = ImGuiDir_Right;
  style.FrameRounding                = 2.0f;
  style.FrameBorderSize              = 1.0f;
  style.GrabRounding                 = 4.0f;
  style.IndentSpacing                = 12.0f;
  style.Colors[ImGuiCol_WindowBg]    = srgb(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_MenuBarBg]   = srgb(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_ScrollbarBg] = srgb(0.2f, 0.2f, 0.2f, 1.0f);
  style.Colors[ImGuiCol_PopupBg]     = srgb(0.135f, 0.135f, 0.135f, 1.0f);
  style.Colors[ImGuiCol_Border]      = srgb(0.4f, 0.4f, 0.4f, 0.5f);
  style.Colors[ImGuiCol_FrameBg]     = srgb(0.05f, 0.05f, 0.05f, 0.5f);

  // Normal
  ImVec4                normal_color = srgb(0.465f, 0.465f, 0.525f, 1.0f);
  std::vector<ImGuiCol> to_change_nrm;
  to_change_nrm.push_back(ImGuiCol_Header);
  to_change_nrm.push_back(ImGuiCol_SliderGrab);
  to_change_nrm.push_back(ImGuiCol_Button);
  to_change_nrm.push_back(ImGuiCol_CheckMark);
  to_change_nrm.push_back(ImGuiCol_ResizeGrip);
  to_change_nrm.push_back(ImGuiCol_TextSelectedBg);
  to_change_nrm.push_back(ImGuiCol_Separator);
  to_change_nrm.push_back(ImGuiCol_FrameBgActive);
  for(auto c : to_change_nrm)
  {
    style.Colors[c] = normal_color;
  }

  // Active
  ImVec4                active_color = srgb(0.365f, 0.365f, 0.425f, 1.0f);
  std::vector<ImGuiCol> to_change_act;
  to_change_act.push_back(ImGuiCol_HeaderActive);
  to_change_act.push_back(ImGuiCol_SliderGrabActive);
  to_change_act.push_back(ImGuiCol_ButtonActive);
  to_change_act.push_back(ImGuiCol_ResizeGripActive);
  to_change_act.push_back(ImGuiCol_SeparatorActive);
  for(auto c : to_change_act)
  {
    style.Colors[c] = active_color;
  }

  // Hovered
  ImVec4                hovered_color = srgb(0.565f, 0.565f, 0.625f, 1.0f);
  std::vector<ImGuiCol> to_change_hover;
  to_change_hover.push_back(ImGuiCol_HeaderHovered);
  to_change_hover.push_back(ImGuiCol_ButtonHovered);
  to_change_hover.push_back(ImGuiCol_FrameBgHovered);
  to_change_hover.push_back(ImGuiCol_ResizeGripHovered);
  to_change_hover.push_back(ImGuiCol_SeparatorHovered);
  for(auto c : to_change_hover)
  {
    style.Colors[c] = hovered_color;
  }


  style.Colors[ImGuiCol_TitleBgActive]    = srgb(0.465f, 0.465f, 0.465f, 1.0f);
  style.Colors[ImGuiCol_TitleBg]          = srgb(0.125f, 0.125f, 0.125f, 1.0f);
  style.Colors[ImGuiCol_Tab]              = srgb(0.05f, 0.05f, 0.05f, 0.5f);
  style.Colors[ImGuiCol_TabHovered]       = srgb(0.465f, 0.495f, 0.525f, 1.0f);
  style.Colors[ImGuiCol_TabActive]        = srgb(0.282f, 0.290f, 0.302f, 1.0f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = srgb(0.465f, 0.465f, 0.465f, 0.350f);

  //Colors_ext[ImGuiColExt_Warning] = srgb (1.0f, 0.43f, 0.35f, 1.0f);

  ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel);
}

//
// Local, return true if the filename exist
//
static bool fileExists(const char* filename)
{
  std::ifstream stream;
  stream.open(filename);
  return stream.is_open();
}

//--------------------------------------------------------------------------------------------------
// Looking for TTF fonts, first on the VULKAN SDK, then Windows default fonts
//
void setFonts(FontMode fontmode)
{
  ImGuiIO&    io             = ImGui::GetIO();
  const float high_dpi_scale = getDPIScale();


  // Nicer fonts
  ImFont* font = nullptr;
  if(fontmode == FONT_MONOSPACED_SCALED)
  {
    if(font == nullptr)
    {
      const std::string p = R"(C:/Windows/Fonts/consola.ttf)";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 12.0f * high_dpi_scale);
    }
    if(font == nullptr)
    {
      const std::string p = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 12.0f * high_dpi_scale);
    }
  }
  else if(fontmode == FONT_PROPORTIONAL_SCALED)
  {
    const char* vk_path = getenv("VK_SDK_PATH");
    if(vk_path)
    {
      const std::string p = std::string(vk_path) + R"(/Samples/Layer-Samples/data/FreeSans.ttf)";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 16.0f * high_dpi_scale);
    }
    if(font == nullptr)
    {
      const std::string p = R"(C:/Windows/Fonts/segoeui.ttf)";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 16.0f * high_dpi_scale);
    }
    if(font == nullptr)
    {
      const std::string p = "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf";
      if(fileExists(p.c_str()))
        font = io.Fonts->AddFontFromFileTTF(p.c_str(), 16.0f * high_dpi_scale);
    }
  }

  if(font == nullptr)
  {
    ImFontConfig font_config = ImFontConfig();
    font_config.SizePixels   = 13.0f * high_dpi_scale;  // 13 is the default font size
    io.Fonts->AddFontDefault(&font_config);
  }
}

void tooltip(const char* description, bool questionMark /*= false*/, float timerThreshold /*= 0.5f*/)
{
  bool passTimer = GImGui->HoveredIdTimer >= timerThreshold && GImGui->ActiveIdTimer == 0.0f;
  if(questionMark)
  {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    passTimer = true;
  }

  if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && passTimer)
  {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(description);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}


// ------------------------------------------------------------------------------------------------

namespace {

template <typename TScalar, ImGuiDataType type, uint8_t dim>
bool show_slider_control_scalar(TScalar* value, TScalar* min, TScalar* max, const char* format)
{
  static const char* visible_labels[] = {"x:", "y:", "z:", "w:"};

  if(dim == 1)
    return ImGui::SliderScalar("##hidden", type, &value[0], &min[0], &max[0], format);

  float indent  = ImGui::GetCursorPos().x;
  bool  changed = false;
  for(uint8_t c = 0; c < dim; ++c)
  {
    ImGui::PushID(c);
    if(c > 0)
    {
      ImGui::NewLine();
      ImGui::SameLine(indent);
    }
    ImGui::Text("%s", visible_labels[c]);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    changed |= ImGui::SliderScalar("##hidden", type, &value[c], &min[c], &max[c], format);
    ImGui::PopID();
  }
  return changed;
}


}  // namespace

template <>
bool Control::show_slider_control<float>(float* value, float& min, float& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 1>(value, &min, &max, format ? format : "%.3f");
}

template <>
bool Control::show_slider_control<glm::vec2>(glm::vec2* value, glm::vec2& min, glm::vec2& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 2>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_slider_control<glm::vec3>(glm::vec3* value, glm::vec3& min, glm::vec3& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 3>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_slider_control<glm::vec4>(glm::vec4* value, glm::vec4& min, glm::vec4& max, const char* format)
{
  return show_slider_control_scalar<float, ImGuiDataType_Float, 4>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<float>(float* value, float speed, float& min, float& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 1>(value, speed, &min, &max, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<glm::vec2>(glm::vec2* value, float speed, glm::vec2& min, glm::vec2& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<glm::vec3>(glm::vec3* value, float speed, glm::vec3& min, glm::vec3& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template <>
bool Control::show_drag_control<glm::vec4>(glm::vec4* value, float speed, glm::vec4& min, glm::vec4& max, const char* format)
{
  return show_drag_control_scalar<float, ImGuiDataType_Float, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}


template <>
bool Control::show_slider_control<int>(int* value, int& min, int& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 1>(value, &min, &max, format ? format : "%d");
}

template <>
bool Control::show_slider_control<glm::ivec2>(glm::ivec2* value, glm::ivec2& min, glm::ivec2& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 2>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_slider_control<glm::ivec3>(glm::ivec3* value, glm::ivec3& min, glm::ivec3& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 3>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_slider_control<glm::ivec4>(glm::ivec4* value, glm::ivec4& min, glm::ivec4& max, const char* format)
{
  return show_slider_control_scalar<int, ImGuiDataType_S32, 4>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_drag_control<int>(int* value, float speed, int& min, int& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 1>(value, speed, &min, &max, format ? format : "%d");
}

template <>
bool Control::show_drag_control<glm::ivec2>(glm::ivec2* value, float speed, glm::ivec2& min, glm::ivec2& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_drag_control<glm::ivec3>(glm::ivec3* value, float speed, glm::ivec3& min, glm::ivec3& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template <>
bool Control::show_drag_control<glm::ivec4>(glm::ivec4* value, float speed, glm::ivec4& min, glm::ivec4& max, const char* format)
{
  return show_drag_control_scalar<int, ImGuiDataType_S32, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}


template <>
bool Control::show_slider_control<uint32_t>(uint32_t* value, uint32_t& min, uint32_t& max, const char* format)
{
  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 1>(value, &min, &max, format ? format : "%d");
}
//
//template <>
//bool Control::show_slider_control<uint32_t_2>(uint32_t_2* value, uint32_t_2& min, uint32_t_2& max, const char* format)
//{
//  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 2>(&value->x, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_slider_control<uint32_t_3>(uint32_t_3* value, uint32_t_3& min, uint32_t_3& max, const char* format)
//{
//  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 3>(&value->x, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_slider_control<uint32_t_4>(uint32_t_4* value, uint32_t_4& min, uint32_t_4& max, const char* format)
//{
//  return show_slider_control_scalar<uint32_t, ImGuiDataType_U32, 4>(&value->x, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t>(uint32_t* value, float speed, uint32_t& min, uint32_t& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 1>(value, speed, &min, &max, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t_2>(uint32_t_2* value, float speed, uint32_t_2& min, uint32_t_2& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t_3>(uint32_t_3* value, float speed, uint32_t_3& min, uint32_t_3& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
//}
//
//template <>
//bool Control::show_drag_control<uint32_t_4>(uint32_t_4* value, float speed, uint32_t_4& min, uint32_t_4& max, const char* format)
//{
//  return show_drag_control_scalar<uint32_t, ImGuiDataType_U32, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
//}


template <>
bool Control::show_slider_control<size_t>(size_t* value, size_t& min, size_t& max, const char* format)
{
  return show_slider_control_scalar<size_t, ImGuiDataType_U64, 1>(value, &min, &max, format ? format : "%d");
}

template <>
bool Control::show_drag_control<size_t>(size_t* value, float speed, size_t& min, size_t& max, const char* format)
{
  return show_drag_control_scalar<size_t, ImGuiDataType_U64, 1>(value, speed, &min, &max, format ? format : "%d");
}

// Static member declaration
ImGuiID Panel::dockspaceID{0};

void Panel::Begin(Side side /*= Side::Right*/, float alpha /*= 0.5f*/, char* name /*= nullptr*/)
{
  // Keeping the unique ID of the dock space
  dockspaceID = ImGui::GetID("DockSpace");

  // The dock need a dummy window covering the entire viewport.
  ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);

  // All flags to dummy window
  ImGuiWindowFlags host_window_flags{};
  host_window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;
  host_window_flags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking;
  host_window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  host_window_flags |= ImGuiWindowFlags_NoBackground;

  // Starting dummy window
  char label[32];
  ImFormatString(label, IM_ARRAYSIZE(label), "DockSpaceViewport_%08X", viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin(label, nullptr, host_window_flags);
  ImGui::PopStyleVar(3);

  // The central node is transparent, so that when UI is draw after, the image is visible
  // Auto Hide Bar, no title of the panel
  // Center is not dockable, that is for the scene
  ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar
                                      | ImGuiDockNodeFlags_NoDockingOverCentralNode;

  // Default panel/window is name setting
  std::string dock_name("Settings");
  if(name != nullptr)
    dock_name = name;

  // Building the splitting of the dock space is done only once
  if(!ImGui::DockBuilderGetNode(dockspaceID))
  {
    ImGui::DockBuilderRemoveNode(dockspaceID);
    ImGui::DockBuilderAddNode(dockspaceID, dockspaceFlags | ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceID, viewport->Size);

    ImGuiID dock_main_id = dockspaceID;

    // Slitting all 4 directions, targetting (320 pixel * DPI) panel width, (180 pixel * DPI) panel height.
    const float xRatio = glm::clamp<float>(320.0f * getDPIScale() / viewport->WorkSize[0], 0.01f, 0.499f);
    const float yRatio = glm::clamp<float>(180.0f * getDPIScale() / viewport->WorkSize[1], 0.01f, 0.499f);
    ImGuiID     id_left, id_right, id_up, id_down;

    // Note, for right, down panels, we use the n / (1 - n) formula to correctly split the space remaining from the left, up panels.
    id_left  = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, xRatio, nullptr, &dock_main_id);
    id_right = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, xRatio / (1 - xRatio), nullptr, &dock_main_id);
    id_up    = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, yRatio, nullptr, &dock_main_id);
    id_down  = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, yRatio / (1 - yRatio), nullptr, &dock_main_id);

    ImGui::DockBuilderDockWindow(side == Side::Left ? dock_name.c_str() : "Dock_left", id_left);
    ImGui::DockBuilderDockWindow(side == Side::Right ? dock_name.c_str() : "Dock_right", id_right);
    ImGui::DockBuilderDockWindow("Dock_up", id_up);
    ImGui::DockBuilderDockWindow("Dock_down", id_down);
    ImGui::DockBuilderDockWindow("Scene", dock_main_id);  // Center

    ImGui::DockBuilderFinish(dock_main_id);
  }

  // Setting the panel to blend with alpha
  ImVec4 col = ImGui::GetStyleColorVec4(ImGuiCol_WindowBg);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(col.x, col.y, col.z, alpha));

  ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), dockspaceFlags);
  ImGui::PopStyleColor();
  ImGui::End();

  // The panel
  if(alpha < 1)
    ImGui::SetNextWindowBgAlpha(alpha);  // For when the panel becomes a floating window
  ImGui::Begin(dock_name.c_str());
}

Control::Style Control::style{};

}  // namespace ImGuiH


bool ImGuiH::azimuthElevationSliders(glm::vec3& direction, bool negative, bool yIsUp /*=true*/)
{
  glm::vec3 normalized_dir = normalize(direction);
  if(negative)
  {
    normalized_dir = -normalized_dir;
  }

  double       azimuth;
  double       elevation;
  const double min_azimuth   = -180.0;
  const double max_azimuth   = 180.0;
  const double min_elevation = -90.0;
  const double max_elevation = 90.0;

  if(yIsUp)
  {
    azimuth   = glm::degrees(atan2(normalized_dir.z, normalized_dir.x));
    elevation = glm::degrees(asin(normalized_dir.y));
  }
  else
  {
    azimuth   = glm::degrees(atan2(normalized_dir.y, normalized_dir.x));
    elevation = glm::degrees(asin(normalized_dir.z));
  }

  namespace PE = ImGuiH::PropertyEditor;
  bool changed = false;
  changed |= PE::SliderScalar("Azimuth", ImGuiDataType_Double, &azimuth, &min_azimuth, &max_azimuth, "%.1f deg",
                              ImGuiSliderFlags_NoRoundToFormat);
  changed |= PE::SliderScalar("Elevation", ImGuiDataType_Double, &elevation, &min_elevation, &max_elevation, "%.1f deg",
                              ImGuiSliderFlags_NoRoundToFormat);

  if(changed)
  {
    azimuth              = glm::radians(azimuth);
    elevation            = glm::radians(elevation);
    double cos_elevation = cos(elevation);

    if(yIsUp)
    {
      direction.y = static_cast<float>(sin(elevation));
      direction.x = static_cast<float>(cos(azimuth) * cos_elevation);
      direction.z = static_cast<float>(sin(azimuth) * cos_elevation);
    }
    else
    {
      direction.z = static_cast<float>(sin(elevation));
      direction.x = static_cast<float>(cos(azimuth) * cos_elevation);
      direction.y = static_cast<float>(sin(azimuth) * cos_elevation);
    }

    if(negative)
    {
      direction = -direction;
    }
  }

  return changed;
}

namespace ImGuiH {
namespace PropertyEditor {

// Beginning the Property Editor
void begin(const char* label, ImGuiTableFlags flag)
{
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
  const bool table_valid = ImGui::BeginTable(label, 2, flag);
  assert(table_valid);
}

// Generic entry, the lambda function should return true if the widget changed
bool entry(const std::string& property_name, const std::function<bool()>& content_fct, const std::string& tooltip)
{
  ImGui::PushID(property_name.c_str());
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::AlignTextToFramePadding();
  ImGui::Text("%s", property_name.c_str());
  if(!tooltip.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_NoSharedDelay))
    ImGui::SetTooltip("%s", tooltip.c_str());
  ImGui::TableNextColumn();
  ImGui::SetNextItemWidth(-FLT_MIN);
  bool result = content_fct();
  ImGui::PopID();
  return result;  // returning if the widget changed
}

// Text specialization
void entry(const std::string& property_name, const std::string& value)
{
  entry(property_name, [&] {
    ImGui::Text("%s", value.c_str());
    return false;  // dummy, no change
  });
}

bool treeNode(const std::string& name)
{
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::AlignTextToFramePadding();
  return ImGui::TreeNodeEx(name.c_str(), ImGuiTreeNodeFlags_SpanFullWidth);
}
void treePop()
{
  ImGui::TreePop();
}

// Ending the Editor
void end()
{
  ImGui::EndTable();
  ImGui::PopStyleVar();
}

bool Button(const char* label, const ImVec2& size, const std::string& tooltip)
{
  return PropertyEditor::entry(
      label, [&] { return ImGui::Button("##hidden", size); }, tooltip);
}
bool SmallButton(const char* label, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SmallButton("##hidden"); }, tooltip);
}
bool Checkbox(const char* label, bool* v, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::Checkbox("##hidden", v); }, tooltip);
}
bool RadioButton(const char* label, bool active, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::RadioButton("##hidden", active); }, tooltip);
}
bool Combo(const char* label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items, const std::string& tooltip)
{
  return entry(label,
               [&] { return ImGui::Combo("##hidden", current_item, items, items_count, popup_max_height_in_items); });
}
bool Combo(const char* label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items, const std::string& tooltip)
{

  return entry(
      label, [&] { return ImGui::Combo("##hidden", current_item, items_separated_by_zeros, popup_max_height_in_items); }, tooltip);
}
bool Combo(const char*        label,
           int*               current_item,
           const char*        (*getter)(void* user_data, int idx),
           void*              user_data,
           int                items_count,
           int                popup_max_height_in_items,
           const std::string& tooltip)
{
  return entry(
      label,
      [&] { return ImGui::Combo("##hidden", current_item, getter, user_data, items_count, popup_max_height_in_items); }, tooltip);
}
bool SliderFloat(const char* label, float* v, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderFloat("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderFloat2(const char* label, float v[2], float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderFloat2("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderFloat3(const char* label, float v[3], float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderFloat3("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderFloat4(const char* label, float v[4], float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderFloat4("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderAngle(const char*        label,
                 float*             v_rad,
                 float              v_degrees_min,
                 float              v_degrees_max,
                 const char*        format,
                 ImGuiSliderFlags   flags,
                 const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderAngle("##hidden", v_rad, v_degrees_min, v_degrees_max, format, flags); }, tooltip);
}
bool SliderInt(const char* label, int* v, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderInt("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderInt2(const char* label, int v[2], int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderInt2("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderInt3(const char* label, int v[3], int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderInt3("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderInt4(const char* label, int v[4], int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderInt4("##hidden", v, v_min, v_max, format, flags); }, tooltip);
}
bool SliderScalar(const char*        label,
                  ImGuiDataType      data_type,
                  void*              p_data,
                  const void*        p_min,
                  const void*        p_max,
                  const char*        format,
                  ImGuiSliderFlags   flags,
                  const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::SliderScalar("##hidden", data_type, p_data, p_min, p_max, format, flags); }, tooltip);
}

bool DragFloat(const char* label, float* v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragFloat("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragFloat2(const char* label, float v[2], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragFloat2("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragFloat3(const char* label, float v[3], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragFloat3("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragFloat4(const char* label, float v[4], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragFloat4("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt(const char* label, int* v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragInt("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt2(const char* label, int v[2], float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragInt2("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt3(const char* label, int v[3], float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragInt3("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragInt4(const char* label, int v[4], float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::DragInt4("##hidden", v, v_speed, v_min, v_max, format, flags); }, tooltip);
}
bool DragScalar(const char*        label,
                ImGuiDataType      data_type,
                void*              p_data,
                float              v_speed /*= 1.0f*/,
                const void*        p_min /*= NULL*/,
                const void*        p_max /*= NULL*/,
                const char*        format /*= NULL*/,
                ImGuiSliderFlags   flags /*= 0*/,
                const std::string& tooltip /*= {}*/)
{
  return entry(
      label, [&] { return ImGui::DragScalar("##hidden", data_type, p_data, v_speed, p_min, p_max, format, flags); }, tooltip);
}
bool InputText(const char* label, char* buf, size_t buf_size, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputText("##hidden", buf, buf_size, flags); }, tooltip);
}
bool InputTextMultiline(const char* label, char* buf, size_t buf_size, const ImVec2& size, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputTextMultiline("##hidden", buf, buf_size, size, flags); }, tooltip);
}
bool InputFloat(const char* label, float* v, float step, float step_fast, const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputFloat("##hidden", v, step, step_fast, format, flags); }, tooltip);
}
bool InputFloat2(const char* label, float v[2], const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputFloat2("##hidden", v, format, flags); }, tooltip);
}
bool InputFloat3(const char* label, float v[3], const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputFloat3("##hidden", v, format, flags); }, tooltip);
}
bool InputFloat4(const char* label, float v[4], const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputFloat4("##hidden", v, format, flags); }, tooltip);
}
bool InputInt(const char* label, int* v, int step, int step_fast, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputInt("##hidden", v, step, step_fast, flags); }, tooltip);
}
bool InputIntClamped(const char* label, int* v, int min, int max, int step, int step_fast, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return Clamped(ImGui::InputInt("##hidden", v, step, step_fast, flags), v, min, max); }, tooltip);
}
bool InputInt2(const char* label, int v[2], ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputInt2("##hidden", v, flags); }, tooltip);
}
bool InputInt3(const char* label, int v[3], ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputInt3("##hidden", v, flags); }, tooltip);
}
bool InputInt4(const char* label, int v[4], ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputInt4("##hidden", v, flags); }, tooltip);
}
bool InputDouble(const char* label, double* v, double step, double step_fast, const char* format, ImGuiInputTextFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::InputDouble("##hidden", v, step, step_fast, format, flags); }, tooltip);
}
bool InputScalar(const char*         label,
                 ImGuiDataType       data_type,
                 void*               p_data,
                 const void*         p_step,
                 const void*         p_step_fast,
                 const char*         format,
                 ImGuiInputTextFlags flags,
                 const std::string&  tooltip)
{
  return entry(
      label, [&] { return ImGui::InputScalar("##hidden", data_type, p_data, p_step, p_step_fast, format, flags); }, tooltip);
}

bool ColorEdit3(const char* label, float col[3], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::ColorEdit3("##hidden", col, flags); }, tooltip);
}
bool ColorEdit4(const char* label, float col[4], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::ColorEdit4("##hidden", col, flags); }, tooltip);
}
bool ColorPicker3(const char* label, float col[3], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::ColorPicker3("##hidden", col, flags); }, tooltip);
}
bool ColorPicker4(const char* label, float col[4], ImGuiColorEditFlags flags, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::ColorPicker4("##hidden", col, flags); }, tooltip);
}
bool ColorButton(const char* label, const ImVec4& col, ImGuiColorEditFlags flags, const ImVec2& size, const std::string& tooltip)
{
  return entry(
      label, [&] { return ImGui::ColorButton("##hidden", col, flags, size); }, tooltip);
}
bool Text(const char* label, const std::string& text)
{
  return entry(label, [&] {
    ImGui::Text("%s", text.c_str());
    return false;  // dummy, no change
  });
}
bool Text(const char* label, const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  bool res = entry(label, [&] {
    ImGui::TextV(fmt, args);
    return false;  // dummy, no change
  });
  va_end(args);
  return res;
}

}  // namespace PropertyEditor
}  // namespace ImGuiH