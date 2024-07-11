/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkDICOMGenericImageFrameInfo.h"
#include "mitkException.h"

mitk::DICOMGenericImageFrameInfo
::DICOMGenericImageFrameInfo(const std::string& filename, unsigned int frameNo)
:DICOMDatasetAccessingImageFrameInfo(filename, frameNo)
{
}

mitk::DICOMGenericImageFrameInfo
::DICOMGenericImageFrameInfo(const DICOMImageFrameInfo::Pointer& frameinfo)
:DICOMDatasetAccessingImageFrameInfo(frameinfo->Filename, frameinfo->FrameNo)
{
}

mitk::DICOMGenericImageFrameInfo::
~DICOMGenericImageFrameInfo()
{
}

mitk::DICOMDatasetFinding
mitk::DICOMGenericImageFrameInfo
::GetTagValueAsString(const DICOMTag& tag) const
{
  DICOMTagPath path(tag);
  DICOMDatasetFinding result;

  const auto finding = m_Values.find(path);
  if (finding != m_Values.cend())
  {
    result.isValid = true;
    result.value = finding->second;
    result.path = path;
  }

  return result;
}

mitk::DICOMDatasetAccess::FindingsListType
mitk::DICOMGenericImageFrameInfo::GetTagValueAsString(const DICOMTagPath& path) const
{
  FindingsListType result;

  for (const auto& iter : m_Values)
  {
    if (path.Equals(iter.first))
    {
      result.emplace_back(true, iter.second, iter.first);
    }
  }

  return result;
}

void
mitk::DICOMGenericImageFrameInfo::SetTagValue(const DICOMTagPath& path, const std::string& value)
{
  if (!path.IsExplicit())
  {
    mitkThrow() << "Only explicit tag paths (no wildcards) are allowed for tag values in DICOMGenericImageFrameInfo. Passed tag path:" << path.ToStr();
  }

  m_Values[path] = value;
}

std::string
mitk::DICOMGenericImageFrameInfo
::GetFilenameIfAvailable() const
{
  return this->Filename;
}
