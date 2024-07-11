/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkRegEvaluationObjectFactory.h"

#include <mitkProperties.h>
#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>

#include "mitkRegEvaluationMapper2D.h"

typedef std::multimap<std::string, std::string> MultimapType;

mitk::RegEvaluationObjectFactory::RegEvaluationObjectFactory()
: CoreObjectFactoryBase()
{
  static bool alreadyDone = false;
  if (!alreadyDone)
  {
    alreadyDone = true;
  }

}

mitk::RegEvaluationObjectFactory::~RegEvaluationObjectFactory()
{
}

mitk::Mapper::Pointer
mitk::RegEvaluationObjectFactory::
CreateMapper(mitk::DataNode* node, MapperSlotId slotId)
{
    mitk::Mapper::Pointer newMapper = nullptr;

    if ( slotId == mitk::BaseRenderer::Standard2D )
    {
        std::string classname("RegEvaluationObject");
        if(node->GetData() && classname.compare(node->GetData()->GetNameOfClass())==0)
        {
          newMapper = mitk::RegEvaluationMapper2D::New();
          newMapper->SetDataNode(node);
        }
    }

    return newMapper;
};

void mitk::RegEvaluationObjectFactory::SetDefaultProperties(mitk::DataNode*)
{

}

std::string mitk::RegEvaluationObjectFactory::GetFileExtensions()
{
  //return empty (dummy) extension string
  return m_FileExtensions.c_str();
};

mitk::CoreObjectFactoryBase::MultimapType mitk::RegEvaluationObjectFactory::GetFileExtensionsMap()
{
  return mitk::CoreObjectFactoryBase::MultimapType();
}

std::string mitk::RegEvaluationObjectFactory::GetSaveFileExtensions()
{
  //return empty (dummy) extension string
  return m_FileExtensions.c_str();
}

mitk::CoreObjectFactoryBase::MultimapType mitk::RegEvaluationObjectFactory::GetSaveFileExtensionsMap()
{
  return mitk::CoreObjectFactoryBase::MultimapType();
}

void mitk::RegEvaluationObjectFactory::RegisterIOFactories()
{
}

struct RegisterRegEvaluationObjectFactoryHelper{
  RegisterRegEvaluationObjectFactoryHelper()
    : m_Factory( mitk::RegEvaluationObjectFactory::New() )
  {
    mitk::CoreObjectFactory::GetInstance()->RegisterExtraFactory( m_Factory );
  }

  ~RegisterRegEvaluationObjectFactoryHelper()
  {
    mitk::CoreObjectFactory::GetInstance()->UnRegisterExtraFactory( m_Factory );
  }

  mitk::RegEvaluationObjectFactory::Pointer m_Factory;
};

static RegisterRegEvaluationObjectFactoryHelper registerMITKRegistrationWrapperIOFactoryHelper;
