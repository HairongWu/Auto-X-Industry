/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include <mitkGIFNeighbouringGreyLevelDependenceFeatures.h>

// MITK
#include <mitkITKImageImport.h>
#include <mitkImageCast.h>
#include <mitkImageAccessByItk.h>

// ITK
#include <itkEnhancedScalarImageToTextureFeaturesFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkNeighborhoodIterator.h>
#include <itkImageRegionConstIterator.h>

// STL
#include <sstream>

struct GIFNeighbouringGreyLevelDependenceFeatureConfiguration
{
  double range;
  unsigned int direction;
  int alpha;

  double MinimumIntensity;
  double MaximumIntensity;
  int Bins;
  mitk::FeatureID id;
};

namespace mitk
{
  struct NGLDMMatrixHolder
  {
  public:
    NGLDMMatrixHolder(double min, double max, int number, int depenence);

    int IntensityToIndex(double intensity);
    double IndexToMinIntensity(int index);
    double IndexToMeanIntensity(int index);
    double IndexToMaxIntensity(int index);

    double m_MinimumRange;
    double m_MaximumRange;
    double m_Stepsize;
    int m_NumberOfDependences;
    int m_NumberOfBins;
    Eigen::MatrixXd m_Matrix;

    int m_NeighbourhoodSize;
    unsigned long m_NumberOfNeighbourVoxels;
    unsigned long m_NumberOfDependenceNeighbourVoxels;
    unsigned long m_NumberOfNeighbourhoods;
    unsigned long m_NumberOfCompleteNeighbourhoods;
  };

  struct NGLDMMatrixFeatures
  {
    NGLDMMatrixFeatures() :
      LowDependenceEmphasis(0),
      HighDependenceEmphasis(0),
      LowGreyLevelCountEmphasis(0),
      HighGreyLevelCountEmphasis(0),
      LowDependenceLowGreyLevelEmphasis(0),
      LowDependenceHighGreyLevelEmphasis(0),
      HighDependenceLowGreyLevelEmphasis(0),
      HighDependenceHighGreyLevelEmphasis(0),
      GreyLevelNonUniformity(0),
      GreyLevelNonUniformityNormalised(0),
      DependenceCountNonUniformity(0),
      DependenceCountNonUniformityNormalised(0),
      DependenceCountPercentage(0),
      GreyLevelVariance(0),
      DependenceCountVariance(0),
      DependenceCountEntropy(0),
      DependenceCountEnergy(0),
      MeanGreyLevelCount(0),
      MeanDependenceCount(0),
      ExpectedNeighbourhoodSize(0),
      AverageNeighbourhoodSize(0),
      AverageIncompleteNeighbourhoodSize(0),
      PercentageOfCompleteNeighbourhoods(0),
      PercentageOfDependenceNeighbours(0)
    {
    }

  public:
    double LowDependenceEmphasis;
    double HighDependenceEmphasis;
    double LowGreyLevelCountEmphasis;
    double HighGreyLevelCountEmphasis;
    double LowDependenceLowGreyLevelEmphasis;
    double LowDependenceHighGreyLevelEmphasis;
    double HighDependenceLowGreyLevelEmphasis;
    double HighDependenceHighGreyLevelEmphasis;

    double GreyLevelNonUniformity;
    double GreyLevelNonUniformityNormalised;
    double DependenceCountNonUniformity;
    double DependenceCountNonUniformityNormalised;

    double DependenceCountPercentage;
    double GreyLevelVariance;
    double DependenceCountVariance;
    double DependenceCountEntropy;
    double DependenceCountEnergy;
    double MeanGreyLevelCount;
    double MeanDependenceCount;

    double ExpectedNeighbourhoodSize;
    double AverageNeighbourhoodSize;
    double AverageIncompleteNeighbourhoodSize;
    double PercentageOfCompleteNeighbourhoods;
    double PercentageOfDependenceNeighbours;

  };
}

static
void MatrixFeaturesTo(const mitk::NGLDMMatrixFeatures& features,
  const GIFNeighbouringGreyLevelDependenceFeatureConfiguration& config,
  mitk::GIFNeighbouringGreyLevelDependenceFeature::FeatureListType& featureList)
{
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Low Dependence Emphasis"), features.LowDependenceEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "High Dependence Emphasis"), features.HighDependenceEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Low Grey Level Count Emphasis"), features.LowGreyLevelCountEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "High Grey Level Count Emphasis"), features.HighGreyLevelCountEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Low Dependence Low Grey Level Emphasis"), features.LowDependenceLowGreyLevelEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Low Dependence High Grey Level Emphasis"), features.LowDependenceHighGreyLevelEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "High Dependence Low Grey Level Emphasis"), features.HighDependenceLowGreyLevelEmphasis));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "High Dependence High Grey Level Emphasis"), features.HighDependenceHighGreyLevelEmphasis));

  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Grey Level Non-Uniformity"), features.GreyLevelNonUniformity));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Grey Level Non-Uniformity Normalised"), features.GreyLevelNonUniformityNormalised));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Non-Uniformity"), features.DependenceCountNonUniformity));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Non-Uniformity Normalised"), features.DependenceCountNonUniformityNormalised));

  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Percentage"), features.DependenceCountPercentage));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Grey Level Mean"), features.MeanGreyLevelCount));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Grey Level Variance"), features.GreyLevelVariance));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Mean"), features.MeanDependenceCount));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Variance"), features.DependenceCountVariance));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Entropy"), features.DependenceCountEntropy));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Dependence Count Energy"), features.DependenceCountEnergy));

  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Expected Neighbourhood Size"), features.ExpectedNeighbourhoodSize));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Average Neighbourhood Size"), features.AverageNeighbourhoodSize));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Average Incomplete Neighbourhood Size"), features.AverageIncompleteNeighbourhoodSize));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Percentage of complete Neighbourhoods"), features.PercentageOfCompleteNeighbourhoods));
  featureList.push_back(std::make_pair(mitk::CreateFeatureID(config.id, "Percentage of Dependence Neighbour Voxels"), features.PercentageOfDependenceNeighbours));
}

mitk::NGLDMMatrixHolder::NGLDMMatrixHolder(double min, double max, int number, int depenence) :
                                            m_MinimumRange(min),
                                            m_MaximumRange(max),
                                            m_Stepsize(0),
                                            m_NumberOfDependences(depenence),
                                            m_NumberOfBins(number),
                                            m_NeighbourhoodSize(1),
                                            m_NumberOfNeighbourVoxels(0),
                                            m_NumberOfDependenceNeighbourVoxels(0),
                                            m_NumberOfNeighbourhoods(0),
                                            m_NumberOfCompleteNeighbourhoods(0)
{
  m_Matrix.resize(number, depenence);
  m_Matrix.fill(0);
  m_Stepsize = (max - min) / (number);
}

int mitk::NGLDMMatrixHolder::IntensityToIndex(double intensity)
{
  return std::floor((intensity - m_MinimumRange) / m_Stepsize);
}

double mitk::NGLDMMatrixHolder::IndexToMinIntensity(int index)
{
  return m_MinimumRange + index * m_Stepsize;
}
double mitk::NGLDMMatrixHolder::IndexToMeanIntensity(int index)
{
  return m_MinimumRange + (index+0.5) * m_Stepsize;
}
double mitk::NGLDMMatrixHolder::IndexToMaxIntensity(int index)
{
  return m_MinimumRange + (index + 1) * m_Stepsize;
}

template<typename TPixel, unsigned int VImageDimension>
void
CalculateNGLDMMatrix(const itk::Image<TPixel, VImageDimension>* itkImage,
                    const itk::Image<unsigned short, VImageDimension>* mask,
                    int alpha,
                    int range,
                    unsigned int direction,
                    mitk::NGLDMMatrixHolder &holder)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef itk::Image<unsigned short, VImageDimension> MaskImageType;
  typedef itk::ConstNeighborhoodIterator<ImageType> ShapeIterType;
  typedef itk::ConstNeighborhoodIterator<MaskImageType> ShapeMaskIterType;

  holder.m_NumberOfCompleteNeighbourhoods = 0;
  holder.m_NumberOfNeighbourhoods = 0;
  holder.m_NumberOfNeighbourVoxels = 0;
  holder.m_NumberOfDependenceNeighbourVoxels = 0;

  itk::Size<VImageDimension> radius;
  radius.Fill(range);

  if ((direction > 1) && (direction - 2 <VImageDimension))
  {
    radius[direction - 2] = 0;
  }

  ShapeIterType imageIter(radius, itkImage, itkImage->GetLargestPossibleRegion());
  ShapeMaskIterType maskIter(radius, mask, mask->GetLargestPossibleRegion());

  auto region = mask->GetLargestPossibleRegion();

  auto center = imageIter.Size() / 2;
  auto iterSize = imageIter.Size();
  holder.m_NeighbourhoodSize = iterSize-1;
  while (!maskIter.IsAtEnd())
  {
    int sameValues = 0;
    bool completeNeighbourhood = true;

    int i = holder.IntensityToIndex(imageIter.GetCenterPixel());

    if ((imageIter.GetCenterPixel() != imageIter.GetCenterPixel()) ||
      (maskIter.GetCenterPixel() < 1))
    {
      ++imageIter;
      ++maskIter;
      continue;
    }

    for (unsigned int position = 0; position < iterSize; ++position)
    {
      if (position == center)
      {
        continue;
      }
      if ( ! region.IsInside(maskIter.GetIndex(position)))
      {
        completeNeighbourhood = false;
        continue;
      }
      bool isInBounds;
      auto jIntensity = imageIter.GetPixel(position, isInBounds);
      auto jMask = maskIter.GetPixel(position, isInBounds);
      if (jMask < 1 || (jIntensity != jIntensity) || ( ! isInBounds))
      {
        completeNeighbourhood = false;
        continue;
      }

      int j = holder.IntensityToIndex(jIntensity);
      holder.m_NumberOfNeighbourVoxels += 1;
      if (std::abs(i - j) <= alpha)
      {
        holder.m_NumberOfDependenceNeighbourVoxels += 1;
        ++sameValues;
      }
    }
    holder.m_Matrix(i, sameValues) += 1;
    holder.m_NumberOfNeighbourhoods += 1;
    if (completeNeighbourhood)
    {
      holder.m_NumberOfCompleteNeighbourhoods += 1;
    }

    ++imageIter;
    ++maskIter;
  }

}

void LocalCalculateFeatures(
  mitk::NGLDMMatrixHolder &holder,
  mitk::NGLDMMatrixFeatures & results
  )
{
  auto sijMatrix = holder.m_Matrix;
  auto piMatrix = holder.m_Matrix;
  auto pjMatrix = holder.m_Matrix;
  // double Ng = holder.m_NumberOfBins;
  // int NgSize = holder.m_NumberOfBins;
  double Ns = sijMatrix.sum();
  piMatrix.rowwise().normalize();
  pjMatrix.colwise().normalize();

  for (int i = 0; i < holder.m_NumberOfBins; ++i)
  {
    double sj = 0;
    for (int j = 0; j < holder.m_NumberOfDependences; ++j)
    {
      double iInt = i+1 ;// holder.IndexToMeanIntensity(i);
      double sij = sijMatrix(i, j);
      double k = j + 1;
      double pij = sij / Ns;

      results.LowDependenceEmphasis += sij / k / k;
      results.HighDependenceEmphasis += sij * k*k;
      if (iInt != 0)
      {
        results.LowGreyLevelCountEmphasis += sij / iInt / iInt;
      }
      results.HighGreyLevelCountEmphasis += sij * iInt*iInt;
      if (iInt != 0)
      {
        results.LowDependenceLowGreyLevelEmphasis += sij / k / k / iInt / iInt;
      }
      results.LowDependenceHighGreyLevelEmphasis += sij * iInt*iInt / k / k;
      if (iInt != 0)
      {
        results.HighDependenceLowGreyLevelEmphasis += sij *k * k / iInt / iInt;
      }
      results.HighDependenceHighGreyLevelEmphasis += sij * k*k*iInt*iInt;


      results.MeanGreyLevelCount += iInt * pij;
      results.MeanDependenceCount += k * pij;
      if (pij > 0)
      {
        results.DependenceCountEntropy -= pij * std::log(pij) / std::log(2);
      }
      results.DependenceCountEnergy += pij*pij;
      sj += sij;
    }
    results.GreyLevelNonUniformity += sj*sj;
    results.GreyLevelNonUniformityNormalised += sj*sj;
  }

  for (int j = 0; j < holder.m_NumberOfDependences; ++j)
  {
    double si = 0;
    for (int i = 0; i < holder.m_NumberOfBins; ++i)
    {
      double sij = sijMatrix(i, j);
      si += sij;
    }
    results.DependenceCountNonUniformity += si*si;
    results.DependenceCountNonUniformityNormalised += si*si;
  }
  for (int i = 0; i < holder.m_NumberOfBins; ++i)
  {
    for (int j = 0; j < holder.m_NumberOfDependences; ++j)
    {
      double iInt = i + 1;// holder.IndexToMeanIntensity(i);
      double sij = sijMatrix(i, j);
      double k = j + 1;
      double pij = sij / Ns;

      results.GreyLevelVariance += (iInt - results.MeanGreyLevelCount)* (iInt - results.MeanGreyLevelCount) * pij;
      results.DependenceCountVariance += (k - results.MeanDependenceCount)* (k - results.MeanDependenceCount) * pij;
    }
  }
  results.LowDependenceEmphasis /= Ns;
  results.HighDependenceEmphasis /= Ns;
  results.LowGreyLevelCountEmphasis /= Ns;
  results.HighGreyLevelCountEmphasis /= Ns;
  results.LowDependenceLowGreyLevelEmphasis /= Ns;
  results.LowDependenceHighGreyLevelEmphasis /= Ns;
  results.HighDependenceLowGreyLevelEmphasis /= Ns;
  results.HighDependenceHighGreyLevelEmphasis /= Ns;

  results.GreyLevelNonUniformity /= Ns;
  results.GreyLevelNonUniformityNormalised /= (Ns*Ns);
  results.DependenceCountNonUniformity /= Ns;
  results.DependenceCountNonUniformityNormalised /= (Ns*Ns);

  results.DependenceCountPercentage = 1;

  results.ExpectedNeighbourhoodSize = holder.m_NeighbourhoodSize;
  results.AverageNeighbourhoodSize = holder.m_NumberOfNeighbourVoxels / (1.0 * holder.m_NumberOfNeighbourhoods);
  results.AverageIncompleteNeighbourhoodSize = (holder.m_NumberOfNeighbourVoxels - holder.m_NumberOfCompleteNeighbourhoods* holder.m_NeighbourhoodSize) / (1.0 * (holder.m_NumberOfNeighbourhoods - holder.m_NumberOfCompleteNeighbourhoods));
  results.PercentageOfCompleteNeighbourhoods = (1.0*holder.m_NumberOfCompleteNeighbourhoods) / (1.0 * holder.m_NumberOfNeighbourhoods);
  results.PercentageOfDependenceNeighbours = holder.m_NumberOfDependenceNeighbourVoxels / (1.0 * holder.m_NumberOfNeighbourVoxels);
}

template<typename TPixel, unsigned int VImageDimension>
void
CalculateCoocurenceFeatures(const itk::Image<TPixel, VImageDimension>* itkImage, const mitk::Image* mask, mitk::GIFNeighbouringGreyLevelDependenceFeature::FeatureListType & featureList, GIFNeighbouringGreyLevelDependenceFeatureConfiguration config)
{
  typedef itk::Image<unsigned short, VImageDimension> MaskType;

  double rangeMin = config.MinimumIntensity;
  double rangeMax = config.MaximumIntensity;
  int numberOfBins = config.Bins;

  typename MaskType::Pointer maskImage = MaskType::New();
  mitk::CastToItkImage(mask, maskImage);

  std::vector<mitk::NGLDMMatrixFeatures> resultVector;
  int numberofDependency = 37;
  if (VImageDimension == 2)
    numberofDependency = 37;

  mitk::NGLDMMatrixHolder holderOverall(rangeMin, rangeMax, numberOfBins, numberofDependency);
  mitk::NGLDMMatrixFeatures overallFeature;
  CalculateNGLDMMatrix<TPixel, VImageDimension>(itkImage, maskImage, config.alpha, config.range, config.direction, holderOverall);
  LocalCalculateFeatures(holderOverall, overallFeature);

  MatrixFeaturesTo(overallFeature, config, featureList);
}

mitk::GIFNeighbouringGreyLevelDependenceFeature::GIFNeighbouringGreyLevelDependenceFeature() :
  m_Ranges({ 1.0 }), m_Alpha(0.)
{
  SetShortName("ngld");
  SetLongName("neighbouring-grey-level-dependence");
  SetFeatureClassName("Neighbouring Grey Level Dependence");
}

void mitk::GIFNeighbouringGreyLevelDependenceFeature::SetRanges(std::vector<double> ranges)
{
  m_Ranges = ranges;
  this->Modified();
}

void mitk::GIFNeighbouringGreyLevelDependenceFeature::SetRange(double range)
{
  m_Ranges.resize(1);
  m_Ranges[0] = range;
  this->Modified();
}

void mitk::GIFNeighbouringGreyLevelDependenceFeature::AddArguments(mitkCommandLineParser& parser) const
{
  AddQuantifierArguments(parser);

  std::string name = GetOptionPrefix();

  parser.addArgument(GetLongName(), name, mitkCommandLineParser::Bool, "Calculate Neighbouring Grey Level Dependence Features", "Calculate Neighbouring grey level dependence based features", us::Any());
  parser.addArgument(name + "::range", name + "::range", mitkCommandLineParser::String, "NGLD Range", "Define the range that is used (Semicolon-separated)", us::Any());
  parser.addArgument(name + "::alpha", name + "::alpha", mitkCommandLineParser::Int, "Int", "", us::Any());
}

mitk::AbstractGlobalImageFeature::FeatureListType mitk::GIFNeighbouringGreyLevelDependenceFeature::DoCalculateFeatures(const Image* image, const Image* mask)
{
  FeatureListType featureList;

  this->InitializeQuantifier(image, mask);
  for (const auto& range : m_Ranges)
  {
    MITK_INFO << "Start calculating NGLD with range " << range << "....";
    GIFNeighbouringGreyLevelDependenceFeatureConfiguration config;
    config.direction = GetDirection();
    config.range = range;
    config.alpha = m_Alpha;

    config.MinimumIntensity = GetQuantifier()->GetMinimum();
    config.MaximumIntensity = GetQuantifier()->GetMaximum();
    config.Bins = GetQuantifier()->GetBins();

    config.id = this->CreateTemplateFeatureID(std::to_string(range), { {GetOptionPrefix() + "::range", range} });

    AccessByItk_3(image, CalculateCoocurenceFeatures, mask, featureList, config);
    MITK_INFO << "Finished calculating NGLD with range " << range << "....";
  }

  return featureList;
}

mitk::AbstractGlobalImageFeature::FeatureListType mitk::GIFNeighbouringGreyLevelDependenceFeature::CalculateFeatures(const Image* image, const Image*, const Image* maskNoNAN)
{
  return Superclass::CalculateFeatures(image, maskNoNAN);
}

std::string mitk::GIFNeighbouringGreyLevelDependenceFeature::GenerateLegacyFeatureEncoding(const FeatureID& id) const
{
  return QuantifierParameterString() + "_Range-" + id.parameters.at(this->GetOptionPrefix() + "::range").ToString();
}

void mitk::GIFNeighbouringGreyLevelDependenceFeature::ConfigureSettingsByParameters(const ParametersType& parameters)
{
  auto prefixname = GetOptionPrefix();

  auto name = prefixname + "::range";
  if (parameters.count(name))
  {
    m_Ranges = SplitDouble(parameters.at(name).ToString(), ';');
  }

  name = prefixname + "::alpha";
  if (parameters.count(name))
  {
    int alpha = us::any_cast<int>(parameters.at(name));
    this->SetAlpha(alpha);
  }
}
