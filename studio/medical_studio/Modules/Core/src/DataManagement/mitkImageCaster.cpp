/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkImageCaster.h"
#include "mitkImageAccessByItk.h"

vtkRenderer *mitk::RendererAccess::m_3DRenderer = nullptr;

void mitk::RendererAccess::Set3DRenderer(vtkRenderer *renwin4)
{
  m_3DRenderer = renwin4;
}

vtkRenderer *mitk::RendererAccess::Get3DRenderer()
{
  return m_3DRenderer;
}

void mitk::Caster::Cast(mitk::BaseData *, mitk::Surface *)
{
}

void mitk::ImageCaster::CastBaseData(mitk::BaseData *const mitkBaseData,
                                     itk::SmartPointer<mitk::Image> &mitkoutputimage)
{
  try
  {
    mitkoutputimage = dynamic_cast<mitk::Image *>(mitkBaseData);
  }
  catch (...)
  {
    return;
  }
}

#define DefineMitkImageCasterMethods(r, data, type)                                                                    \
  void mitk::ImageCaster::CastToItkImage(const mitk::Image *mitkImage,                                                 \
                                         itk::SmartPointer<itk::Image<BOOST_PP_TUPLE_REM(2) type>> &itkOutputImage)     \
  {                                                                                                                    \
    mitk::CastToItkImage(mitkImage, itkOutputImage);                                                                   \
  }                                                                                                                    \
  void mitk::ImageCaster::CastToMitkImage(const itk::Image<BOOST_PP_TUPLE_REM(2) type> *itkImage,                       \
                                          itk::SmartPointer<mitk::Image> &mitkOutputImage)                             \
  {                                                                                                                    \
    mitk::CastToMitkImage<itk::Image<BOOST_PP_TUPLE_REM(2) type>>(itkImage, mitkOutputImage);                           \
  }

BOOST_PP_SEQ_FOR_EACH(DefineMitkImageCasterMethods, _, MITK_ACCESSBYITK_TYPES_DIMN_SEQ(2))
BOOST_PP_SEQ_FOR_EACH(DefineMitkImageCasterMethods, _, MITK_ACCESSBYITK_TYPES_DIMN_SEQ(3))
