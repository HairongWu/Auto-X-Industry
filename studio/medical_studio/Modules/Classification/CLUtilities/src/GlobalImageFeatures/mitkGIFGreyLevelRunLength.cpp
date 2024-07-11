/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkGIFGreyLevelRunLength.h>

// MITK
#include <mitkITKImageImport.h>
#include <mitkImageCast.h>
#include <mitkImageAccessByItk.h>

// ITK
#include <itkEnhancedScalarImageToRunLengthFeaturesFilter.h>

// STL
#include <sstream>

namespace mitk
{
  struct GIFGreyLevelRunLengthParameters
  {
    unsigned int m_Direction;

    double MinimumIntensity;
    double MaximumIntensity;
    int Bins;
    FeatureID id;
  };
}


template<typename TPixel, unsigned int VImageDimension>
void
  CalculateGrayLevelRunLengthFeatures(const itk::Image<TPixel, VImageDimension>* itkImage, const mitk::Image* mask, mitk::GIFGreyLevelRunLength::FeatureListType & featureList, mitk::GIFGreyLevelRunLengthParameters params)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef itk::Image<TPixel, VImageDimension> MaskType;
  typedef itk::Statistics::EnhancedScalarImageToRunLengthFeaturesFilter<ImageType> FilterType;
  typedef typename FilterType::RunLengthFeaturesFilterType TextureFilterType;

  typename MaskType::Pointer maskImage = MaskType::New();
  mitk::CastToItkImage(mask, maskImage);

  typename FilterType::Pointer filter = FilterType::New();
  typename FilterType::Pointer filter2 = FilterType::New();

  typename FilterType::OffsetVector::Pointer newOffset = FilterType::OffsetVector::New();
  auto oldOffsets = filter->GetOffsets();
  auto oldOffsetsIterator = oldOffsets->Begin();
  while (oldOffsetsIterator != oldOffsets->End())
  {
    bool continueOuterLoop = false;
    typename FilterType::OffsetType offset = oldOffsetsIterator->Value();
    for (unsigned int i = 0; i < VImageDimension; ++i)
    {
      if (params.m_Direction == i + 2 && offset[i] != 0)
      {
        continueOuterLoop = true;
      }
    }
    if (params.m_Direction == 1)
    {
      offset[0] = 0;
      offset[1] = 0;
      offset[2] = 1;
      newOffset->push_back(offset);
      break;
    }

    oldOffsetsIterator++;
    if (continueOuterLoop)
      continue;
    newOffset->push_back(offset);
  }
  filter->SetOffsets(newOffset);
  filter2->SetOffsets(newOffset);


  // All features are required
  typename FilterType::FeatureNameVectorPointer requestedFeatures = FilterType::FeatureNameVector::New();
  requestedFeatures->push_back(TextureFilterType::ShortRunEmphasis);
  requestedFeatures->push_back(TextureFilterType::LongRunEmphasis);
  requestedFeatures->push_back(TextureFilterType::GreyLevelNonuniformity);
  requestedFeatures->push_back(TextureFilterType::GreyLevelNonuniformityNormalized);
  requestedFeatures->push_back(TextureFilterType::RunLengthNonuniformity);
  requestedFeatures->push_back(TextureFilterType::RunLengthNonuniformityNormalized);
  requestedFeatures->push_back(TextureFilterType::LowGreyLevelRunEmphasis);
  requestedFeatures->push_back(TextureFilterType::HighGreyLevelRunEmphasis);
  requestedFeatures->push_back(TextureFilterType::ShortRunLowGreyLevelEmphasis);
  requestedFeatures->push_back(TextureFilterType::ShortRunHighGreyLevelEmphasis);
  requestedFeatures->push_back(TextureFilterType::LongRunLowGreyLevelEmphasis);
  requestedFeatures->push_back(TextureFilterType::LongRunHighGreyLevelEmphasis);
  requestedFeatures->push_back(TextureFilterType::RunPercentage);
  requestedFeatures->push_back(TextureFilterType::NumberOfRuns);
  requestedFeatures->push_back(TextureFilterType::GreyLevelVariance);
  requestedFeatures->push_back(TextureFilterType::RunLengthVariance);
  requestedFeatures->push_back(TextureFilterType::RunEntropy);

  filter->SetInput(itkImage);
  filter->SetMaskImage(maskImage);
  filter->SetRequestedFeatures(requestedFeatures);
  filter2->SetInput(itkImage);
  filter2->SetMaskImage(maskImage);
  filter2->SetRequestedFeatures(requestedFeatures);
  int numberOfBins = params.Bins;
  if (numberOfBins < 2)
    numberOfBins = 256;

  double minRange = params.MinimumIntensity;
  double maxRange = params.MaximumIntensity;

  filter->SetPixelValueMinMax(minRange, maxRange);
  filter->SetNumberOfBinsPerAxis(numberOfBins);
  filter2->SetPixelValueMinMax(minRange, maxRange);
  filter2->SetNumberOfBinsPerAxis(numberOfBins);

  filter->SetDistanceValueMinMax(0, numberOfBins);
  filter2->SetDistanceValueMinMax(0, numberOfBins);

  filter2->CombinedFeatureCalculationOn();

  filter->Update();
  filter2->Update();

  auto featureMeans = filter->GetFeatureMeans ();
  auto featureStd = filter->GetFeatureStandardDeviations();
  auto featureCombined = filter2->GetFeatureMeans();

  for (std::size_t i = 0; i < featureMeans->size(); ++i)
  {
    switch (i)
    {
    case TextureFilterType::ShortRunEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::LongRunEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::GreyLevelNonuniformity :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level nonuniformity Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level nonuniformity Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level nonuniformity Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::GreyLevelNonuniformityNormalized :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level nonuniformity normalized Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level nonuniformity normalized Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level nonuniformity normalized Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::RunLengthNonuniformity :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length nonuniformity Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length nonuniformity Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length nonuniformity Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::RunLengthNonuniformityNormalized :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length nonuniformity normalized Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length nonuniformity normalized Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length nonuniformity normalized Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::LowGreyLevelRunEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Low grey level run emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Low grey level run emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Low grey level run emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::HighGreyLevelRunEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "High grey level run emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "High grey level run emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "High grey level run emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::ShortRunLowGreyLevelEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run low grey level emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run low grey level emphasis  Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run low grey level emphasis  Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::ShortRunHighGreyLevelEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run high grey level emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run high grey level emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Short run high grey level emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::LongRunLowGreyLevelEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run low grey level emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run low grey level emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run low grey level emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::LongRunHighGreyLevelEmphasis :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run high grey level emphasis Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run high grey level emphasis Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Long run high grey level emphasis Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::RunPercentage :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run percentage Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run percentage Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run percentage Comb."), featureCombined->ElementAt(i) / newOffset->size()));
      break;
    case TextureFilterType::NumberOfRuns :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Number of runs Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Number of runs Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Number of runs Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::GreyLevelVariance :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level variance Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level variance Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Grey level variance Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::RunLengthVariance :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length variance Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length variance Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length variance Comb."), featureCombined->ElementAt(i)));
      break;
    case TextureFilterType::RunEntropy :
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length entropy Means"), featureMeans->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length entropy Std."), featureStd->ElementAt(i)));
      featureList.push_back(std::make_pair(mitk::CreateFeatureID(params.id, "Run length entropy Comb."), featureCombined->ElementAt(i)));
      break;
    default:
      break;
    }
  }
}

mitk::GIFGreyLevelRunLength::GIFGreyLevelRunLength()
{
  SetShortName("rl");
  SetLongName("run-length");
  SetFeatureClassName("Run Length");
}

void mitk::GIFGreyLevelRunLength::AddArguments(mitkCommandLineParser &parser) const
{
  this->AddQuantifierArguments(parser);

  std::string name = GetOptionPrefix();

  parser.addArgument(GetLongName(), name, mitkCommandLineParser::Bool, "Use Run-Length", "Calculates Run-Length based features", us::Any());
}

mitk::AbstractGlobalImageFeature::FeatureListType mitk::GIFGreyLevelRunLength::DoCalculateFeatures(const Image* image, const Image* mask)
{
  FeatureListType featureList;

  InitializeQuantifier(image, mask);

  MITK_INFO << "Start calculating Run-length";

  GIFGreyLevelRunLengthParameters params;

  params.m_Direction = GetDirection();

  params.MinimumIntensity = GetQuantifier()->GetMinimum();
  params.MaximumIntensity = GetQuantifier()->GetMaximum();
  params.Bins = GetQuantifier()->GetBins();
  params.id = this->CreateTemplateFeatureID();


  MITK_INFO << params.MinimumIntensity;
  MITK_INFO << params.MaximumIntensity;
  MITK_INFO << params.m_Direction;
  MITK_INFO << params.Bins;

  AccessByItk_3(image, CalculateGrayLevelRunLengthFeatures, mask, featureList, params);

  MITK_INFO << "Finished calculating Run-length";
  
  return featureList;
}

mitk::AbstractGlobalImageFeature::FeatureListType mitk::GIFGreyLevelRunLength::CalculateFeatures(const Image* image, const Image*, const Image* maskNoNAN)
{
  return Superclass::CalculateFeatures(image, maskNoNAN);
}
