/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         https://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#ifndef __itkEnhancedScalarImageToSizeZoneFeaturesFilter_hxx
#define __itkEnhancedScalarImageToSizeZoneFeaturesFilter_hxx

#include "itkEnhancedScalarImageToSizeZoneFeaturesFilter.h"
#include "itkNeighborhood.h"
#include <itkImageRegionConstIterator.h>
#include "vnl/vnl_math.h"

namespace itk
{
  namespace Statistics
  {
    template<typename TImage, typename THistogramFrequencyContainer>
    EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::EnhancedScalarImageToSizeZoneFeaturesFilter()
    {
      this->SetNumberOfRequiredInputs( 1 );
      this->SetNumberOfRequiredOutputs( 1 );

      for( int i = 0; i < 2; ++i )
      {
        this->ProcessObject::SetNthOutput( i, this->MakeOutput( i ) );
      }

      this->m_SizeZoneMatrixGenerator = SizeZoneMatrixFilterType::New();
      this->m_FeatureMeans = FeatureValueVector::New();
      this->m_FeatureStandardDeviations = FeatureValueVector::New();

      // Set the requested features to the default value:
      // {Energy, Entropy, InverseDifferenceMoment, Inertia, ClusterShade,
      // ClusterProminence}
      FeatureNameVectorPointer requestedFeatures = FeatureNameVector::New();
      // can't directly set this->m_RequestedFeatures since it is const!

      requestedFeatures->push_back( SizeZoneFeaturesFilterType::SmallZoneEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::LargeZoneEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::GreyLevelNonuniformity );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::GreyLevelNonuniformityNormalized );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::SizeZoneNonuniformity );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::SizeZoneNonuniformityNormalized );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::LowGreyLevelZoneEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::HighGreyLevelZoneEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::SmallZoneLowGreyLevelEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::SmallZoneHighGreyLevelEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::LargeZoneLowGreyLevelEmphasis );
      requestedFeatures->push_back( SizeZoneFeaturesFilterType::LargeZoneHighGreyLevelEmphasis );
      requestedFeatures->push_back( 20 );

      this->SetRequestedFeatures( requestedFeatures );

      // Set the offset directions to their defaults: half of all the possible
      // directions 1 pixel away. (The other half is included by symmetry.)
      // We use a neighborhood iterator to calculate the appropriate offsets.
      typedef Neighborhood<typename ImageType::PixelType,
        ImageType::ImageDimension> NeighborhoodType;
      NeighborhoodType hood;
      hood.SetRadius( 1 );

      // select all "previous" neighbors that are face+edge+vertex
      // connected to the current pixel. do not include the center pixel.
      unsigned int centerIndex = hood.GetCenterNeighborhoodIndex();
      OffsetVectorPointer offsets = OffsetVector::New();
      for( unsigned int d = 0; d < centerIndex; d++ )
      {
        OffsetType offset = hood.GetOffset( d );
        offsets->push_back( offset );
      }
      this->SetOffsets( offsets );
      this->m_FastCalculations = false;
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    typename
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::DataObjectPointer
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::MakeOutput( DataObjectPointerArraySizeType itkNotUsed(idx) )
    {
      return FeatureValueVectorDataObjectType::New().GetPointer();
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::GenerateData(void)
    {
      if ( this->m_FastCalculations )
      {
        this->FastCompute();
      }
      else
      {
        this->FullCompute();
      }
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::FullCompute()
    {
      int numOffsets = this->m_Offsets->size();
      int numFeatures = this->m_RequestedFeatures->size();
      double **features;

      features = new double *[numOffsets];
      for( int i = 0; i < numOffsets; i++ )
      {
        features[i] = new double[numFeatures];
      }

      unsigned long numberOfVoxels = 0;
      ImageRegionConstIterator<TImage> voxelCountIter(this->GetMaskImage(),this->GetMaskImage()->GetLargestPossibleRegion());
      while ( ! voxelCountIter.IsAtEnd() )
      {
        if (voxelCountIter.Get() > 0)
          ++numberOfVoxels;
        ++voxelCountIter;
      }

      // For each offset, calculate each feature
      typename OffsetVector::ConstIterator offsetIt;
      int offsetNum, featureNum;
      typedef typename SizeZoneFeaturesFilterType::SizeZoneFeatureName
        InternalSizeZoneFeatureName;

      for( offsetIt = this->m_Offsets->Begin(), offsetNum = 0;
        offsetIt != this->m_Offsets->End(); offsetIt++, offsetNum++ )
      {
        this->m_SizeZoneMatrixGenerator->SetOffset( offsetIt.Value() );
        this->m_SizeZoneMatrixGenerator->Update();
        typename SizeZoneFeaturesFilterType::Pointer SizeZoneMatrixCalculator =
          SizeZoneFeaturesFilterType::New();
        SizeZoneMatrixCalculator->SetInput(
          this->m_SizeZoneMatrixGenerator->GetOutput() );
        SizeZoneMatrixCalculator->SetNumberOfVoxels(numberOfVoxels);
        SizeZoneMatrixCalculator->Update();

        typename FeatureNameVector::ConstIterator fnameIt;
        for( fnameIt = this->m_RequestedFeatures->Begin(), featureNum = 0;
          fnameIt != this->m_RequestedFeatures->End(); fnameIt++, featureNum++ )
        {
          features[offsetNum][featureNum] = SizeZoneMatrixCalculator->GetFeature(
            ( InternalSizeZoneFeatureName )fnameIt.Value() );
        }
      }

      // Now get the mean and deviaton of each feature across the offsets.
      this->m_FeatureMeans->clear();
      this->m_FeatureStandardDeviations->clear();
      double *tempFeatureMeans = new double[numFeatures];
      double *tempFeatureDevs = new double[numFeatures];

      /*Compute incremental mean and SD, a la Knuth, "The  Art of Computer
      Programming, Volume 2: Seminumerical Algorithms",  section 4.2.2.
      Compute mean and standard deviation using the recurrence relation:
      M(1) = x(1), M(k) = M(k-1) + (x(k) - M(k-1) ) / k
      S(1) = 0, S(k) = S(k-1) + (x(k) - M(k-1)) * (x(k) - M(k))
      for 2 <= k <= n, then
      sigma = std::sqrt(S(n) / n) (or divide by n-1 for sample SD instead of
      population SD).
      */

      // Set up the initial conditions (k = 1)
      for( featureNum = 0; featureNum < numFeatures; featureNum++ )
      {
        tempFeatureMeans[featureNum] = features[0][featureNum];
        tempFeatureDevs[featureNum] = 0;
      }
      // Zone through the recurrence (k = 2 ... N)
      for( offsetNum = 1; offsetNum < numOffsets; offsetNum++ )
      {
        int k = offsetNum + 1;
        for( featureNum = 0; featureNum < numFeatures; featureNum++ )
        {
          double M_k_minus_1 = tempFeatureMeans[featureNum];
          double S_k_minus_1 = tempFeatureDevs[featureNum];
          double x_k = features[offsetNum][featureNum];

          double M_k = M_k_minus_1 + ( x_k - M_k_minus_1 ) / k;
          double S_k = S_k_minus_1 + ( x_k - M_k_minus_1 ) * ( x_k - M_k );

          tempFeatureMeans[featureNum] = M_k;
          tempFeatureDevs[featureNum] = S_k;
        }
      }
      for( featureNum = 0; featureNum < numFeatures; featureNum++ )
      {
        tempFeatureDevs[featureNum] = std::sqrt( tempFeatureDevs[featureNum] /
          numOffsets );

        this->m_FeatureMeans->push_back( tempFeatureMeans[featureNum] );
        this->m_FeatureStandardDeviations->push_back( tempFeatureDevs[featureNum] );
      }

      FeatureValueVectorDataObjectType *meanOutputObject =
        itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>( this->ProcessObject::GetOutput( 0 ) );
      meanOutputObject->Set( this->m_FeatureMeans );

      FeatureValueVectorDataObjectType *standardDeviationOutputObject =
        itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>( this->ProcessObject::GetOutput( 1 ) );
      standardDeviationOutputObject->Set( this->m_FeatureStandardDeviations );

      delete[] tempFeatureMeans;
      delete[] tempFeatureDevs;
      for( int i = 0; i < numOffsets; i++ )
      {
        delete[] features[i];
      }
      delete[] features;
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::FastCompute()
    {
      // Compute the feature for the first offset
      typename OffsetVector::ConstIterator offsetIt = this->m_Offsets->Begin();
      this->m_SizeZoneMatrixGenerator->SetOffset( offsetIt.Value() );

      this->m_SizeZoneMatrixGenerator->Update();
      typename SizeZoneFeaturesFilterType::Pointer SizeZoneMatrixCalculator =
        SizeZoneFeaturesFilterType::New();
      SizeZoneMatrixCalculator->SetInput(
        this->m_SizeZoneMatrixGenerator->GetOutput() );
      SizeZoneMatrixCalculator->Update();

      typedef typename SizeZoneFeaturesFilterType::SizeZoneFeatureName
        InternalSizeZoneFeatureName;
      this->m_FeatureMeans->clear();
      this->m_FeatureStandardDeviations->clear();
      typename FeatureNameVector::ConstIterator fnameIt;
      for( fnameIt = this->m_RequestedFeatures->Begin();
        fnameIt != this->m_RequestedFeatures->End(); fnameIt++ )
      {
        this->m_FeatureMeans->push_back( SizeZoneMatrixCalculator->GetFeature(
          ( InternalSizeZoneFeatureName )fnameIt.Value() ) );
        this->m_FeatureStandardDeviations->push_back( 0.0 );
      }

      FeatureValueVectorDataObjectType *meanOutputObject =
        itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>( this->ProcessObject::GetOutput( 0 ) );
      meanOutputObject->Set( this->m_FeatureMeans );

      FeatureValueVectorDataObjectType *standardDeviationOutputObject =
        itkDynamicCastInDebugMode<FeatureValueVectorDataObjectType *>( this->ProcessObject::GetOutput( 1 ) );
      standardDeviationOutputObject->Set( this->m_FeatureStandardDeviations );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::SetInput( const ImageType *image )
    {
      // Process object is not const-correct so the const_cast is required here
      this->ProcessObject::SetNthInput( 0,
        const_cast<ImageType *>( image ) );

      this->m_SizeZoneMatrixGenerator->SetInput( image );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::SetNumberOfBinsPerAxis( unsigned int numberOfBins )
    {
      itkDebugMacro( "setting NumberOfBinsPerAxis to " << numberOfBins );
      this->m_SizeZoneMatrixGenerator->SetNumberOfBinsPerAxis( numberOfBins );
      this->Modified();
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::SetPixelValueMinMax( PixelType min, PixelType max )
    {
      itkDebugMacro( "setting Min to " << min << "and Max to " << max );
      this->m_SizeZoneMatrixGenerator->SetPixelValueMinMax( min, max );
      this->Modified();
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::SetDistanceValueMinMax( double min, double max )
    {
      itkDebugMacro( "setting Min to " << min << "and Max to " << max );
      this->m_SizeZoneMatrixGenerator->SetDistanceValueMinMax( min, max );
      this->Modified();
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::SetMaskImage( const ImageType *image )
    {
      // Process object is not const-correct so the const_cast is required here
      this->ProcessObject::SetNthInput( 1,
        const_cast< ImageType * >( image ) );

      this->m_SizeZoneMatrixGenerator->SetMaskImage( image );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    const TImage *
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::GetInput() const
    {
      if ( this->GetNumberOfInputs() < 1 )
      {
        return ITK_NULLPTR;
      }
      return static_cast<const ImageType *>( this->ProcessObject::GetInput( 0 ) );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    const typename
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::FeatureValueVectorDataObjectType *
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::GetFeatureMeansOutput() const
    {
      return itkDynamicCastInDebugMode<const FeatureValueVectorDataObjectType *>
        (this->ProcessObject::GetOutput( 0 ) );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    const typename
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::FeatureValueVectorDataObjectType *
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::GetFeatureStandardDeviationsOutput() const
    {
      return itkDynamicCastInDebugMode< const FeatureValueVectorDataObjectType * >
        ( this->ProcessObject::GetOutput( 1 ) );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    const TImage *
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::GetMaskImage() const
    {
      if ( this->GetNumberOfInputs() < 2 )
      {
        return ITK_NULLPTR;
      }
      return static_cast< const ImageType *>( this->ProcessObject::GetInput( 1 ) );
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::SetInsidePixelValue( PixelType insidePixelValue )
    {
      itkDebugMacro( "setting InsidePixelValue to " << insidePixelValue );
      this->m_SizeZoneMatrixGenerator->SetInsidePixelValue( insidePixelValue );
      this->Modified();
    }

    template<typename TImage, typename THistogramFrequencyContainer>
    void
      EnhancedScalarImageToSizeZoneFeaturesFilter<TImage, THistogramFrequencyContainer>
      ::PrintSelf(std::ostream & os, Indent indent) const
    {
      Superclass::PrintSelf(os, indent);
      os << indent << "RequestedFeatures: "
        << this->GetRequestedFeatures() << std::endl;
      os << indent << "FeatureStandardDeviations: "
        << this->GetFeatureStandardDeviations() << std::endl;
      os << indent << "FastCalculations: "
        << this->GetFastCalculations() << std::endl;
      os << indent << "Offsets: " << this->GetOffsets() << std::endl;
      os << indent << "FeatureMeans: " << this->GetFeatureMeans() << std::endl;
    }
  } // end of namespace Statistics
} // end of namespace itk

#endif
