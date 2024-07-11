/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkDICOMGDCMTagScanner.h"
#include "mitkDICOMGDCMTagCache.h"
#include "mitkDICOMGDCMImageFrameInfo.h"

#include <gdcmScanner.h>

mitk::DICOMGDCMTagScanner::DICOMGDCMTagScanner()
{
  m_GDCMScanner = std::make_shared<gdcm::Scanner>();
}

mitk::DICOMGDCMTagScanner::~DICOMGDCMTagScanner()
{
}

mitk::DICOMDatasetFinding mitk::DICOMGDCMTagScanner::GetTagValue( DICOMImageFrameInfo* frame, const DICOMTag& tag ) const
{
  assert( frame );
  assert(m_Cache.IsNotNull());

  if (m_Cache.IsNull())
  {
    mitkThrow() << "Wrong usage of DICOMGDCMScanner- Called GetTagValue() before scanned at least once. No scanner cache available.";
  }

  return m_Cache->GetTagValue(frame, tag);
}

void mitk::DICOMGDCMTagScanner::AddTag( const DICOMTag& tag )
{
  m_ScannedTags.insert( tag );
  m_GDCMScanner->AddTag(
    gdcm::Tag( tag.GetGroup(), tag.GetElement() ) ); // also a set, duplicate calls to AddTag don't hurt
}

void mitk::DICOMGDCMTagScanner::AddTags( const DICOMTagList& tags )
{
  for ( auto tagIter = tags.cbegin(); tagIter != tags.cend(); ++tagIter )
  {
    this->AddTag( *tagIter );
  }
}

void mitk::DICOMGDCMTagScanner::AddTagPath(const DICOMTagPath& path)
{
  if (path.Size() != 1 || !path.IsExplicit())
  {
    std::stringstream errorstring;
    errorstring << "Invalid call to DICOMGDCMTagScanner::AddTagPath(). "
      << "Scanner does only support paths that are explicitly specify one tag. "
      << "Invalid path: "<<path.ToStr();
    MITK_ERROR << errorstring.str();
    throw std::invalid_argument(errorstring.str());
  }
  this->AddTag(path.GetFirstNode().tag);
}

void mitk::DICOMGDCMTagScanner::AddTagPaths(const DICOMTagPathList& paths)
{
  for (const auto& path : paths)
  {
    if (path.Size() != 1 || !path.IsExplicit())
    {
      std::stringstream errorstring;
      errorstring << "Invalid call to DICOMGDCMTagScanner::AddTagPaths(). "
        << "Scanner does only support paths that are explicitly specify one tag. "
        << "Invalid path: " << path.ToStr();
      MITK_ERROR << errorstring.str();
      throw std::invalid_argument(errorstring.str());
    }
    this->AddTag(path.GetFirstNode().tag);
  }
}

void mitk::DICOMGDCMTagScanner::SetInputFiles( const StringList& filenames )
{
  m_InputFilenames = filenames;
}


void mitk::DICOMGDCMTagScanner::Scan()
{
  // TODO integrate push/pop locale??
  m_GDCMScanner->Scan( m_InputFilenames );

  DICOMGDCMTagCache::Pointer newCache = DICOMGDCMTagCache::New();
  newCache->InitCache(m_ScannedTags, m_GDCMScanner, m_InputFilenames);

  m_Cache = newCache;
}

mitk::DICOMTagCache::Pointer
mitk::DICOMGDCMTagScanner::GetScanCache() const
{
  return m_Cache.GetPointer();
}

mitk::DICOMDatasetAccessingImageFrameList mitk::DICOMGDCMTagScanner::GetFrameInfoList() const
{
  mitk::DICOMDatasetAccessingImageFrameList result;
  if (m_Cache.IsNotNull())
  {
    result = m_Cache->GetFrameInfoList();
  }
  return result;
}
