/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkApplyDiffImageOperation.h"

#include <itkCommand.h>

mitk::ApplyDiffImageOperation::ApplyDiffImageOperation(OperationType operationType,
                                                       Image *image,
                                                       Image *diffImage,
                                                       unsigned int timeStep,
                                                       unsigned int sliceDimension,
                                                       unsigned int sliceIndex)
  : Operation(operationType),
    m_Image(image),
    m_SliceIndex(sliceIndex),
    m_SliceDimension(sliceDimension),
    m_TimeStep(timeStep),
    m_Factor(1.0),
    m_ImageStillValid(false),
    m_DeleteTag(0)
{
  if (image && diffImage)
  {
    // observe 3D image for DeleteEvent
    m_ImageStillValid = true;

    itk::SimpleMemberCommand<ApplyDiffImageOperation>::Pointer command =
      itk::SimpleMemberCommand<ApplyDiffImageOperation>::New();
    command->SetCallbackFunction(this, &ApplyDiffImageOperation::OnImageDeleted);
    m_DeleteTag = image->AddObserver(itk::DeleteEvent(), command);

    // keep a compressed version of the image
    m_CompressedImageContainer.CompressImage(diffImage);
  }
}

mitk::ApplyDiffImageOperation::~ApplyDiffImageOperation()
{
  if (m_ImageStillValid)
  {
    m_Image->RemoveObserver(m_DeleteTag);
  }
}

void mitk::ApplyDiffImageOperation::OnImageDeleted()
{
  m_ImageStillValid = false;
}

mitk::Image::Pointer mitk::ApplyDiffImageOperation::GetDiffImage()
{
  // uncompress image to create a valid mitk::Image
  return m_CompressedImageContainer.DecompressImage();
}
