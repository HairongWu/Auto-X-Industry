/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#include <usModuleActivator.h>
#include <usModuleContext.h>

#include <mitkModelFitProviderBase.h>

//general models
#include "mitkGenericParamModelFactory.h"
#include "mitkLinearModelFactory.h"
#include "mitkExponentialDecayModelFactory.h"
#include "mitkExpDecayOffsetModelFactory.h"
#include "mitkExponentialSaturationModelFactory.h"
#include "mitkTwoStepLinearModelFactory.h"
#include "mitkThreeStepLinearModelFactory.h"

namespace mitk
{

/*
 * This is the module activator for the IO aspects of the "ModelFit" module.
 */
class ModelFitModelsActivator : public us::ModuleActivator
{
public:

  template <class TProvider> void RegisterProvider(us::ModuleContext* context)
  {
    auto provider = new TProvider();
    provider->RegisterService(context);

    m_RegisteredProviders.push_back(std::unique_ptr<IModelFitProvider>(provider));
  }

  void Load(us::ModuleContext* context) override
  {
    m_RegisteredProviders.clear();
    RegisterProvider<mitk::ModelFitProviderBase<mitk::GenericParamModelFactory> >(context);
    RegisterProvider<mitk::ModelFitProviderBase<mitk::LinearModelFactory> >(context);
    RegisterProvider<mitk::ModelFitProviderBase<mitk::ExponentialDecayModelFactory> >(context);
    RegisterProvider<mitk::ModelFitProviderBase<mitk::ExpDecayOffsetModelFactory> >(context);
    RegisterProvider<mitk::ModelFitProviderBase<mitk::ExponentialSaturationModelFactory> >(context);
    RegisterProvider<mitk::ModelFitProviderBase<mitk::TwoStepLinearModelFactory> >(context);
    RegisterProvider<mitk::ModelFitProviderBase<mitk::ThreeStepLinearModelFactory> >(context);
  }

  void Unload(us::ModuleContext* ) override
  {
  }

private:
  std::vector<std::unique_ptr<IModelFitProvider> > m_RegisteredProviders;
};

}

US_EXPORT_MODULE_ACTIVATOR(mitk::ModelFitModelsActivator)
