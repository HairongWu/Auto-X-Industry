/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkIOUtil.h"
#include "mitkImage.h"
#include "mitkTestingMacros.h"
#include "mitkVolumeCalculator.h"
#include <mitkStandaloneDataStorage.h>

int mitkVolumeCalculatorTest(int /*argc*/, char *argv[])
{
  MITK_TEST_BEGIN("VolumeCalculator")
  const char *filename = argv[1];
  const char *filename3D = argv[2];
  mitk::VolumeCalculator::Pointer volumeCalculator = mitk::VolumeCalculator::New();
  //*********************************************************************
  // Part I: Testing calculated volume.
  // The correct values have been manually calculated using external software.
  //*********************************************************************

  mitk::Image::Pointer image = mitk::IOUtil::Load<mitk::Image>(filename);
  MITK_TEST_CONDITION_REQUIRED(image.IsNotNull(), "01 Check if test image could be loaded");

  volumeCalculator->SetImage(image);
  volumeCalculator->SetThreshold(0);
  volumeCalculator->ComputeVolume();
  float volume = volumeCalculator->GetVolume();

  MITK_TEST_CONDITION_REQUIRED(volume == 1600, "02 Test Volume Result. Expected 1600 actual value " << volume);

  volumeCalculator->SetThreshold(255);
  volumeCalculator->ComputeVolume();
  volume = volumeCalculator->GetVolume();

  MITK_TEST_CONDITION_REQUIRED(volume == 1272.50, "03 Test Volume Result. Expected 1272.50 actual value " << volume);

  image = mitk::IOUtil::Load<mitk::Image>(filename3D);

  volumeCalculator->SetImage(image);
  volumeCalculator->SetThreshold(-1023);
  volumeCalculator->ComputeVolume();
  std::vector<float> volumes = volumeCalculator->GetVolumes();

  for (auto it = volumes.begin(); it != volumes.end(); ++it)
  {
    MITK_TEST_CONDITION_REQUIRED((*it) == 24.576f, "04 Test Volume Result.");
  }
  MITK_TEST_END()
}
