/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkCoreObjectFactory.h"
#include "mitkConfig.h"

#include "mitkColorProperty.h"
#include "mitkDataNode.h"
#include "mitkEnumerationProperty.h"
#include "mitkGeometry3D.h"
#include "mitkGeometryData.h"
#include "mitkImage.h"
#include "mitkLevelWindowProperty.h"
#include "mitkLookupTable.h"
#include "mitkLookupTableProperty.h"
#include "mitkPlaneGeometry.h"
#include "mitkPlaneGeometryData.h"
#include "mitkPlaneGeometryDataMapper2D.h"
#include "mitkPlaneGeometryDataVtkMapper3D.h"
#include "mitkPointSet.h"
#include "mitkPointSetVtkMapper2D.h"
#include "mitkPointSetVtkMapper3D.h"
#include "mitkProperties.h"
#include "mitkPropertyList.h"
#include "mitkSlicedGeometry3D.h"
#include "mitkSmartPointerProperty.h"
#include "mitkStringProperty.h"
#include "mitkSurface.h"
#include "mitkSurface.h"
#include "mitkSurfaceVtkMapper2D.h"
#include "mitkSurfaceVtkMapper3D.h"
#include "mitkTimeGeometry.h"
#include "mitkTransferFunctionProperty.h"
#include "mitkVtkInterpolationProperty.h"
#include "mitkVtkRepresentationProperty.h"
#include "mitkVtkResliceInterpolationProperty.h"
#include <mitkImageVtkMapper2D.h>

// Legacy Support:
#include <mitkCoreServices.h>
#include <mitkLegacyFileReaderService.h>
#include <mitkLegacyFileWriterService.h>

#include <mitkCrosshairData.h>
#include <mitkCrosshairVtkMapper2D.h>

void mitk::CoreObjectFactory::RegisterExtraFactory(CoreObjectFactoryBase *factory)
{
  MITK_DEBUG << "CoreObjectFactory: registering extra factory of type " << factory->GetNameOfClass();
  m_ExtraFactories.insert(CoreObjectFactoryBase::Pointer(factory));
  // Register Legacy Reader and Writer
  this->RegisterLegacyReaders(factory);
  this->RegisterLegacyWriters(factory);
}

void mitk::CoreObjectFactory::UnRegisterExtraFactory(CoreObjectFactoryBase *factory)
{
  MITK_DEBUG << "CoreObjectFactory: un-registering extra factory of type " << factory->GetNameOfClass();
  this->UnRegisterLegacyWriters(factory);
  this->UnRegisterLegacyReaders(factory);
  try
  {
    m_ExtraFactories.erase(factory);
  }
  catch ( const std::exception &e )
  {
    MITK_ERROR << "Caught exception while unregistering: " << e.what();
  }
}

mitk::CoreObjectFactory::Pointer mitk::CoreObjectFactory::GetInstance()
{
  static mitk::CoreObjectFactory::Pointer instance;
  if (instance.IsNull())
  {
    instance = mitk::CoreObjectFactory::New();
  }
  return instance;
}

mitk::CoreObjectFactory::~CoreObjectFactory()
{
  for (auto iter =
         m_LegacyReaders.begin();
       iter != m_LegacyReaders.end();
       ++iter)
  {
    for (auto &elem : iter->second)
    {
      delete elem;
    }
  }

  for (auto iter =
         m_LegacyWriters.begin();
       iter != m_LegacyWriters.end();
       ++iter)
  {
    for (auto &elem : iter->second)
    {
      delete elem;
    }
  }
}

void mitk::CoreObjectFactory::SetDefaultProperties(mitk::DataNode *node)
{
  if (node == nullptr)
    return;

  mitk::DataNode::Pointer nodePointer = node;

  mitk::Image* image = dynamic_cast<mitk::Image *>(node->GetData());
  if (nullptr != image && image->IsInitialized())
  {
    mitk::ImageVtkMapper2D::SetDefaultProperties(node);
  }

  if (nullptr != dynamic_cast<mitk::PlaneGeometryData*>(node->GetData()))
  {
    mitk::PlaneGeometryDataMapper2D::SetDefaultProperties(node);
  }

  if (nullptr != dynamic_cast<mitk::Surface*>(node->GetData()))
  {
    mitk::SurfaceVtkMapper2D::SetDefaultProperties(node);
    mitk::SurfaceVtkMapper3D::SetDefaultProperties(node);
  }

  if (nullptr != dynamic_cast<mitk::PointSet*>(node->GetData()))
  {
    mitk::PointSetVtkMapper2D::SetDefaultProperties(node);
    mitk::PointSetVtkMapper3D::SetDefaultProperties(node);
  }

  if (nullptr != dynamic_cast<mitk::CrosshairData*>(node->GetData()))
  {
    mitk::CrosshairVtkMapper2D::SetDefaultProperties(node);
  }

  for (auto it = m_ExtraFactories.begin(); it != m_ExtraFactories.end(); ++it)
  {
    (*it)->SetDefaultProperties(node);
  }
}

mitk::CoreObjectFactory::CoreObjectFactory()
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    CreateFileExtensionsMap();

    // RegisterLegacyReaders(this);
    // RegisterLegacyWriters(this);

    alreadyDone = true;
  }
}

mitk::Mapper::Pointer mitk::CoreObjectFactory::CreateMapper(mitk::DataNode *node, MapperSlotId id)
{
  mitk::Mapper::Pointer newMapper = nullptr;
  mitk::Mapper::Pointer tmpMapper = nullptr;

  // check whether extra factories provide mapper
  for (auto it = m_ExtraFactories.begin(); it != m_ExtraFactories.end(); ++it)
  {
    tmpMapper = (*it)->CreateMapper(node, id);
    if (tmpMapper.IsNotNull())
      newMapper = tmpMapper;
  }

  if (newMapper.IsNull())
  {
    mitk::BaseData *data = node->GetData();

    if (id == mitk::BaseRenderer::Standard2D)
    {
      if ((dynamic_cast<Image *>(data) != nullptr))
      {
        newMapper = mitk::ImageVtkMapper2D::New();
        newMapper->SetDataNode(node);
      }
      else if ((dynamic_cast<PlaneGeometryData *>(data) != nullptr))
      {
        newMapper = mitk::PlaneGeometryDataMapper2D::New();
        newMapper->SetDataNode(node);
      }
      else if ((dynamic_cast<Surface *>(data) != nullptr))
      {
        newMapper = mitk::SurfaceVtkMapper2D::New();
        // cast because SetDataNode is not virtual
        auto *castedMapper = dynamic_cast<mitk::SurfaceVtkMapper2D *>(newMapper.GetPointer());
        castedMapper->SetDataNode(node);
      }
      else if ((dynamic_cast<PointSet *>(data) != nullptr))
      {
        newMapper = mitk::PointSetVtkMapper2D::New();
        newMapper->SetDataNode(node);
      }
    }
    else if (id == mitk::BaseRenderer::Standard3D)
    {
      if ((dynamic_cast<PlaneGeometryData *>(data) != nullptr))
      {
        newMapper = mitk::PlaneGeometryDataVtkMapper3D::New();
        newMapper->SetDataNode(node);
      }
      else if ((dynamic_cast<Surface *>(data) != nullptr))
      {
        newMapper = mitk::SurfaceVtkMapper3D::New();
        newMapper->SetDataNode(node);
      }
      else if ((dynamic_cast<PointSet *>(data) != nullptr))
      {
        newMapper = mitk::PointSetVtkMapper3D::New();
        newMapper->SetDataNode(node);
      }
    }
  }

  return newMapper;
}

std::string mitk::CoreObjectFactory::GetFileExtensions()
{
  MultimapType aMap;
  for (auto it = m_ExtraFactories.begin(); it != m_ExtraFactories.end(); ++it)
  {
    aMap = (*it)->GetFileExtensionsMap();
    this->MergeFileExtensions(m_FileExtensionsMap, aMap);
  }
  this->CreateFileExtensions(m_FileExtensionsMap, m_FileExtensions);
  return m_FileExtensions.c_str();
}

void mitk::CoreObjectFactory::MergeFileExtensions(MultimapType &fileExtensionsMap, MultimapType inputMap)
{
  std::pair<MultimapType::iterator, MultimapType::iterator> pairOfIter;
  for (auto it = inputMap.begin(); it != inputMap.end(); ++it)
  {
    bool duplicateFound = false;
    pairOfIter = fileExtensionsMap.equal_range((*it).first);
    for (auto it2 = pairOfIter.first; it2 != pairOfIter.second; ++it2)
    {
      // cout << "  [" << (*it).first << ", " << (*it).second << "]" << endl;
      std::string aString = (*it2).second;
      if (aString.compare((*it).second) == 0)
      {
        // cout << "  DUP!! [" << (*it).first << ", " << (*it).second << "]" << endl;
        duplicateFound = true;
        break;
      }
    }
    if (!duplicateFound)
    {
      fileExtensionsMap.insert(std::pair<std::string, std::string>((*it).first, (*it).second));
    }
  }
}

mitk::CoreObjectFactoryBase::MultimapType mitk::CoreObjectFactory::GetFileExtensionsMap()
{
  return m_FileExtensionsMap;
}

void mitk::CoreObjectFactory::CreateFileExtensionsMap()
{
  /*
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.dcm", "DICOM files"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.DCM", "DICOM files"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.dc3", "DICOM files"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.DC3", "DICOM files"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.gdcm", "DICOM files"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.seq", "DKFZ Pic"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.seq.gz", "DKFZ Pic"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.dcm", "Sets of 2D slices"));
  m_FileExtensionsMap.insert(std::pair<std::string, std::string>("*.gdcm", "Sets of 2D slices"));
  */
}

std::string mitk::CoreObjectFactory::GetSaveFileExtensions()
{
  MultimapType aMap;
  for (auto it = m_ExtraFactories.begin(); it != m_ExtraFactories.end(); ++it)
  {
    aMap = (*it)->GetSaveFileExtensionsMap();
    this->MergeFileExtensions(m_SaveFileExtensionsMap, aMap);
  }
  this->CreateFileExtensions(m_SaveFileExtensionsMap, m_SaveFileExtensions);
  return m_SaveFileExtensions.c_str();
}

mitk::CoreObjectFactoryBase::MultimapType mitk::CoreObjectFactory::GetSaveFileExtensionsMap()
{
  return m_SaveFileExtensionsMap;
}

mitk::CoreObjectFactory::FileWriterList mitk::CoreObjectFactory::GetFileWriters()
{
  FileWriterList allWriters = m_FileWriters;
  // sort to merge lists later on
  typedef std::set<mitk::FileWriterWithInformation::Pointer> FileWriterSet;
  FileWriterSet fileWritersSet;

  fileWritersSet.insert(allWriters.begin(), allWriters.end());

  // collect all extra factories
  for (auto it = m_ExtraFactories.begin(); it != m_ExtraFactories.end(); ++it)
  {
    FileWriterList list2 = (*it)->GetFileWriters();

    // add them to the sorted set
    fileWritersSet.insert(list2.begin(), list2.end());
  }

  // write back to allWriters to return a list
  allWriters.clear();
  allWriters.insert(allWriters.end(), fileWritersSet.begin(), fileWritersSet.end());

  return allWriters;
}

void mitk::CoreObjectFactory::MapEvent(const mitk::Event *, const int)
{
}

std::string mitk::CoreObjectFactory::GetDescriptionForExtension(const std::string &extension)
{
  std::multimap<std::string, std::string> fileExtensionMap = GetSaveFileExtensionsMap();
  for (auto it = fileExtensionMap.begin(); it != fileExtensionMap.end();
       ++it)
    if (it->first == extension)
      return it->second;
  return ""; // If no matching extension was found, return empty string
}

void mitk::CoreObjectFactory::RegisterLegacyReaders(mitk::CoreObjectFactoryBase *factory)
{
  // We are not really interested in the string, just call the method since
  // many readers initialize the map the first time when this method is called
  factory->GetFileExtensions();

  std::map<std::string, std::vector<std::string>> extensionsByCategories;
  std::multimap<std::string, std::string> fileExtensionMap = factory->GetFileExtensionsMap();
  for (auto it = fileExtensionMap.begin(); it != fileExtensionMap.end();
       ++it)
  {
    std::string extension = it->first;
    // remove "*."
    extension = extension.erase(0, 2);

    extensionsByCategories[it->second].push_back(extension);
  }

  for (auto &extensionsByCategorie : extensionsByCategories)
  {
    m_LegacyReaders[factory].push_back(
      new mitk::LegacyFileReaderService(extensionsByCategorie.second, extensionsByCategorie.first));
  }
}

void mitk::CoreObjectFactory::UnRegisterLegacyReaders(mitk::CoreObjectFactoryBase *factory)
{
  auto iter =
    m_LegacyReaders.find(factory);
  if (iter != m_LegacyReaders.end())
  {
    for (auto &elem : iter->second)
    {
      delete elem;
    }

    m_LegacyReaders.erase(iter);
  }
}

void mitk::CoreObjectFactory::RegisterLegacyWriters(mitk::CoreObjectFactoryBase *factory)
{
  // Get all external Writers
  mitk::CoreObjectFactory::FileWriterList writers = factory->GetFileWriters();

  // We are not really interested in the string, just call the method since
  // many writers initialize the map the first time when this method is called
  factory->GetSaveFileExtensions();

  MultimapType fileExtensionMap = factory->GetSaveFileExtensionsMap();

  for (auto it = writers.begin(); it != writers.end(); ++it)
  {
    std::vector<std::string> extensions = (*it)->GetPossibleFileExtensions();
    if (extensions.empty())
      continue;

    std::string description;
    for (auto ext = extensions.begin(); ext != extensions.end(); ++ext)
    {
      if (ext->empty())
        continue;

      std::string extension = *ext;
      std::string extensionWithStar = extension;
      if (extension.find_first_of('*') == 0)
      {
        // remove "*."
        extension = extension.substr(0, extension.size() - 2);
      }
      else
      {
        extensionWithStar.insert(extensionWithStar.begin(), '*');
      }

      for (auto fileExtensionIter = fileExtensionMap.begin();
           fileExtensionIter != fileExtensionMap.end();
           ++fileExtensionIter)
      {
        if (fileExtensionIter->first == extensionWithStar)
        {
          description = fileExtensionIter->second;
          break;
        }
      }
      if (!description.empty())
        break;
    }
    if (description.empty())
    {
      description = std::string("Legacy ") + (*it)->GetNameOfClass() + " Reader";
    }

    mitk::FileWriter::Pointer fileWriter(it->GetPointer());
    mitk::LegacyFileWriterService *lfws = new mitk::LegacyFileWriterService(fileWriter, description);
    m_LegacyWriters[factory].push_back(lfws);
  }
}

void mitk::CoreObjectFactory::UnRegisterLegacyWriters(mitk::CoreObjectFactoryBase *factory)
{
  auto iter =
    m_LegacyWriters.find(factory);
  if (iter != m_LegacyWriters.end())
  {
    for (auto &elem : iter->second)
    {
      delete elem;
    }

    m_LegacyWriters.erase(iter);
  }
}
