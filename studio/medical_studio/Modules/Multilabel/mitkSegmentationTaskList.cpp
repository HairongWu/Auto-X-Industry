/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkSegmentationTaskList.h"

#include <mitkIOUtil.h>
#include <mitkProperties.h>

mitk::SegmentationTaskList::Task::Task()
  : m_Defaults(nullptr)
{
}

mitk::SegmentationTaskList::Task::~Task()
{
}

void mitk::SegmentationTaskList::Task::SetDefaults(const Task* defaults)
{
  m_Defaults = defaults;
}

mitk::SegmentationTaskList::SegmentationTaskList()
{
  // A base data cannot be serialized if empty. To be not considered empty its
  // geometry must consist of at least one time step. However, a segmentation
  // task would then appear as invisible spatial object in a scene. This can
  // be prevented by excluding it from the scene's bounding box calculations.
  this->GetTimeGeometry()->Expand(1);
  this->SetProperty("includeInBoundingBox", BoolProperty::New(false));
}

mitk::SegmentationTaskList::SegmentationTaskList(const Self& other)
  : BaseData(other)
{
}

mitk::SegmentationTaskList::~SegmentationTaskList()
{
}

size_t mitk::SegmentationTaskList::GetNumberOfTasks() const
{
  return m_Tasks.size();
}

size_t mitk::SegmentationTaskList::AddTask(const Task& subtask)
{
  m_Tasks.push_back(subtask);
  m_Tasks.back().SetDefaults(&m_Defaults);
  return m_Tasks.size() - 1;
}

const mitk::SegmentationTaskList::Task* mitk::SegmentationTaskList::GetTask(size_t index) const
{
  return &m_Tasks.at(index);
}

mitk::SegmentationTaskList::Task* mitk::SegmentationTaskList::GetTask(size_t index)
{
  return &m_Tasks.at(index);
}

const mitk::SegmentationTaskList::Task& mitk::SegmentationTaskList::GetDefaults() const
{
  return m_Defaults;
}

void mitk::SegmentationTaskList::SetDefaults(const Task& defaults)
{
  m_Defaults = defaults;

  for (auto& subtask : m_Tasks)
    subtask.SetDefaults(&m_Defaults);
}

bool mitk::SegmentationTaskList::IsDone() const
{
  for (size_t i = 0; i < m_Tasks.size(); ++i)
  {
    if (!this->IsDone(i))
      return false;
  }

  return true;
}

bool mitk::SegmentationTaskList::IsDone(size_t index) const
{
  return fs::exists(this->GetAbsolutePath(m_Tasks.at(index).GetResult()));
}

fs::path mitk::SegmentationTaskList::GetInputLocation() const
{
  std::string inputLocation;
  this->GetPropertyList()->GetStringProperty("MITK.IO.reader.inputlocation", inputLocation);

  return !inputLocation.empty()
#ifdef MITK_HAS_FILESYSTEM
    ? fs::path(inputLocation).lexically_normal()
#else
    ? fs::path(inputLocation)
#endif
    : fs::path();
}

fs::path mitk::SegmentationTaskList::GetBasePath() const
{
  return this->GetInputLocation().remove_filename();
}

fs::path mitk::SegmentationTaskList::GetAbsolutePath(const fs::path& path) const
{
  if (path.empty())
    return path;

#ifdef MITK_HAS_FILESYSTEM
  auto normalizedPath = path.lexically_normal();
#else
  auto normalizedPath = path;
#endif

  return !normalizedPath.is_absolute()
    ? this->GetBasePath() / normalizedPath
    : normalizedPath;
}

fs::path mitk::SegmentationTaskList::GetInterimPath(const fs::path& path) const
{
  if (path.empty() || !path.has_filename())
    return path;

  auto interimPath = path;
  return interimPath.replace_extension(".interim" + path.extension().string());
}

void mitk::SegmentationTaskList::SaveTask(size_t index, const BaseData* segmentation, bool saveAsInterimResult)
{
  if (segmentation == nullptr)
    return;

  auto path = this->GetAbsolutePath(this->GetResult(index));
  auto interimPath = this->GetInterimPath(path);

  if (fs::exists(path))
    saveAsInterimResult = false;

  IOUtil::Save(segmentation, saveAsInterimResult
    ? interimPath.string()
    : path.string());

  if (!saveAsInterimResult && fs::exists(interimPath))
  {
    std::error_code ec;
    fs::remove(interimPath, ec);
  }
}

std::vector<mitk::SegmentationTaskList::Task>::const_iterator mitk::SegmentationTaskList::begin() const
{
  return m_Tasks.begin();
}

std::vector<mitk::SegmentationTaskList::Task>::const_iterator mitk::SegmentationTaskList::end() const
{
  return m_Tasks.end();
}

std::vector<mitk::SegmentationTaskList::Task>::iterator mitk::SegmentationTaskList::begin()
{
  return m_Tasks.begin();
}

std::vector<mitk::SegmentationTaskList::Task>::iterator mitk::SegmentationTaskList::end()
{
  return m_Tasks.end();
}

void mitk::SegmentationTaskList::SetRequestedRegionToLargestPossibleRegion()
{
}

bool mitk::SegmentationTaskList::RequestedRegionIsOutsideOfTheBufferedRegion()
{
  return false;
}

bool mitk::SegmentationTaskList::VerifyRequestedRegion()
{
  return true;
}

void mitk::SegmentationTaskList::SetRequestedRegion(const itk::DataObject*)
{
}
