/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleActivator.h>
#include <usModuleContext.h>
#include <usServiceEvent.h>

#include "mitkDICOMPMIO.h"

#include "mitkDICOMPMIOMimeTypes.h"

namespace mitk
{
  /**
  \brief Registers services for multilabel dicom module.
  */
  class DICOMPMIOActivator : public us::ModuleActivator
  {
    std::vector<AbstractFileIO *> m_FileIOs;

  public:
    void Load(us::ModuleContext * context) override
    {
      us::ServiceProperties props;
      props[us::ServiceConstants::SERVICE_RANKING()] = 10;

      std::vector<mitk::CustomMimeType *> mimeTypes = mitk::MitkDICOMPMIOMimeTypes::Get();
      for (std::vector<mitk::CustomMimeType *>::const_iterator mimeTypeIter = mimeTypes.begin(),
        iterEnd = mimeTypes.end();
        mimeTypeIter != iterEnd;
        ++mimeTypeIter)
      {
        context->RegisterService(*mimeTypeIter, props);
      }
      m_FileIOs.push_back(new DICOMPMIO());
    }
    void Unload(us::ModuleContext *) override
    {
      for (auto &elem : m_FileIOs)
      {
        delete elem;
      }
    }
  };
}

US_EXPORT_MODULE_ACTIVATOR(mitk::DICOMPMIOActivator)
