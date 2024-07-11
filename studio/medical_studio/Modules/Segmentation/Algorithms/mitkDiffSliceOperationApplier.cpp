/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkDiffSliceOperationApplier.h"

#include "mitkDiffSliceOperation.h"
#include "mitkRenderingManager.h"
#include "mitkSegTool2D.h"
#include <mitkExtractSliceFilter.h>
#include <mitkVtkImageOverwrite.h>

// VTK
#include <vtkSmartPointer.h>

mitk::DiffSliceOperationApplier::DiffSliceOperationApplier()
{
}

mitk::DiffSliceOperationApplier::~DiffSliceOperationApplier()
{
}

void mitk::DiffSliceOperationApplier::ExecuteOperation(Operation *operation)
{
  auto *imageOperation = dynamic_cast<DiffSliceOperation *>(operation);

  // as we only support DiffSliceOperation return if operation is not type of DiffSliceOperation
  if (!imageOperation)
    return;

  // check if the operation is valid
  if (imageOperation->IsValid())
  {
    // the actual overwrite filter (vtk)
    vtkSmartPointer<mitkVtkImageOverwrite> reslice = vtkSmartPointer<mitkVtkImageOverwrite>::New();

    mitk::Image::Pointer slice = imageOperation->GetSlice();
    // Set the slice as 'input'
    reslice->SetInputSlice(slice->GetVtkImageData());

    // set overwrite mode to true to write back to the image volume
    reslice->SetOverwriteMode(true);
    reslice->Modified();

    // a wrapper for vtkImageOverwrite
    mitk::ExtractSliceFilter::Pointer extractor = mitk::ExtractSliceFilter::New(reslice);
    extractor->SetInput(imageOperation->GetImage());
    extractor->SetTimeStep(imageOperation->GetTimeStep());
    extractor->SetWorldGeometry(dynamic_cast<const PlaneGeometry *>(imageOperation->GetWorldGeometry()));
    extractor->SetVtkOutputRequest(true);
    extractor->SetResliceTransformByGeometry(imageOperation->GetImage()->GetGeometry(imageOperation->GetTimeStep()));

    extractor->Modified();
    extractor->Update();

    // make sure the modification is rendered
    RenderingManager::GetInstance()->RequestUpdateAll();
    imageOperation->GetImage()->Modified();

    PlaneGeometry::ConstPointer plane = dynamic_cast<const PlaneGeometry *>(imageOperation->GetWorldGeometry());
    SegTool2D::UpdateAllSurfaceInterpolations(dynamic_cast<LabelSetImage*>(imageOperation->GetImage()), imageOperation->GetTimeStep(), plane, true);
  }
}

mitk::DiffSliceOperationApplier *mitk::DiffSliceOperationApplier::GetInstance()
{
  static auto *s_Instance = new DiffSliceOperationApplier();
  return s_Instance;
}
