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
#ifndef __itkMaskedStatisticsImageFilter_hxx
#define __itkMaskedStatisticsImageFilter_hxx
#include "itkMaskedStatisticsImageFilter.h"


#include "itkImageScanlineIterator.h"
#include "itkProgressReporter.h"

namespace itk
{
  template< typename TInputImage, typename TMaskImage >
  MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::MaskedStatisticsImageFilter():m_ThreadSum(1), m_SumOfSquares(1), m_Count(1), m_ThreadMin(1), m_ThreadMax(1)
  {
    this->DynamicMultiThreadingOff();

    // first output is a copy of the image, DataObject created by
    // superclass
    //
    // allocate the data objects for the outputs which are
    // just decorators around pixel types
    for ( int i = 1; i < 3; ++i )
    {
      typename PixelObjectType::Pointer output =
        static_cast< PixelObjectType * >( this->MakeOutput(i).GetPointer() );
      this->ProcessObject::SetNthOutput( i, output.GetPointer() );
    }
    // allocate the data objects for the outputs which are
    // just decorators around real types
    for ( int i = 3; i < 7; ++i )
    {
      typename RealObjectType::Pointer output =
        static_cast< RealObjectType * >( this->MakeOutput(i).GetPointer() );
      this->ProcessObject::SetNthOutput( i, output.GetPointer() );
    }

    this->GetMinimumOutput()->Set( NumericTraits< PixelType >::max() );
    this->GetMaximumOutput()->Set( NumericTraits< PixelType >::NonpositiveMin() );
    this->GetMeanOutput()->Set( NumericTraits< RealType >::max() );
    this->GetSigmaOutput()->Set( NumericTraits< RealType >::max() );
    this->GetVarianceOutput()->Set( NumericTraits< RealType >::max() );
    this->GetSumOutput()->Set(NumericTraits< RealType >::Zero);
  }

  template< typename TInputImage, typename TMaskImage >
  DataObject::Pointer
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::MakeOutput(DataObjectPointerArraySizeType output)
  {
    switch ( output )
    {
    case 0:
      return TInputImage::New().GetPointer();
      break;
    case 1:
      return PixelObjectType::New().GetPointer();
      break;
    case 2:
      return PixelObjectType::New().GetPointer();
      break;
    case 3:
    case 4:
    case 5:
    case 6:
      return RealObjectType::New().GetPointer();
      break;
    default:
      // might as well make an image
      return TInputImage::New().GetPointer();
      break;
    }
  }

  template< typename TInputImage, typename TMaskImage >
  typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::PixelObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetMinimumOutput()
  {
    return static_cast< PixelObjectType * >( this->ProcessObject::GetOutput(1) );
  }

  template< typename TInputImage, typename TMaskImage >
  const typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::PixelObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetMinimumOutput() const
  {
    return static_cast< const PixelObjectType * >( this->ProcessObject::GetOutput(1) );
  }

  template< typename TInputImage, typename TMaskImage >
  typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::PixelObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetMaximumOutput()
  {
    return static_cast< PixelObjectType * >( this->ProcessObject::GetOutput(2) );
  }

  template< typename TInputImage, typename TMaskImage >
  const typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::PixelObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetMaximumOutput() const
  {
    return static_cast< const PixelObjectType * >( this->ProcessObject::GetOutput(2) );
  }

  template< typename TInputImage, typename TMaskImage >
  typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetMeanOutput()
  {
    return static_cast< RealObjectType * >( this->ProcessObject::GetOutput(3) );
  }

  template< typename TInputImage, typename TMaskImage >
  const typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetMeanOutput() const
  {
    return static_cast< const RealObjectType * >( this->ProcessObject::GetOutput(3) );
  }

  template< typename TInputImage, typename TMaskImage >
  typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetSigmaOutput()
  {
    return static_cast< RealObjectType * >( this->ProcessObject::GetOutput(4) );
  }

  template< typename TInputImage, typename TMaskImage >
  const typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetSigmaOutput() const
  {
    return static_cast< const RealObjectType * >( this->ProcessObject::GetOutput(4) );
  }

  template< typename TInputImage, typename TMaskImage >
  typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetVarianceOutput()
  {
    return static_cast< RealObjectType * >( this->ProcessObject::GetOutput(5) );
  }

  template< typename TInputImage, typename TMaskImage >
  const typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetVarianceOutput() const
  {
    return static_cast< const RealObjectType * >( this->ProcessObject::GetOutput(5) );
  }

  template< typename TInputImage, typename TMaskImage >
  typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetSumOutput()
  {
    return static_cast< RealObjectType * >( this->ProcessObject::GetOutput(6) );
  }

  template< typename TInputImage, typename TMaskImage >
  const typename MaskedStatisticsImageFilter< TInputImage, TMaskImage >::RealObjectType *
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GetSumOutput() const
  {
    return static_cast< const RealObjectType * >( this->ProcessObject::GetOutput(6) );
  }

  template< typename TInputImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::GenerateInputRequestedRegion()
  {
    Superclass::GenerateInputRequestedRegion();
    if ( this->GetInput() )
    {
      InputImagePointer image =
        const_cast< typename Superclass::InputImageType * >( this->GetInput() );
      image->SetRequestedRegionToLargestPossibleRegion();
    }
  }

  template< typename TInputImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::EnlargeOutputRequestedRegion(DataObject *data)
  {
    Superclass::EnlargeOutputRequestedRegion(data);
    data->SetRequestedRegionToLargestPossibleRegion();
  }

  template< typename TInputImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::AllocateOutputs()
  {
    // Pass the input through as the output
    InputImagePointer image =
      const_cast< TInputImage * >( this->GetInput() );

    this->GraftOutput(image);

    // Nothing that needs to be allocated for the remaining outputs
  }

  template< typename TInputImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::BeforeThreadedGenerateData()
  {
    ThreadIdType numberOfThreads = this->GetNumberOfWorkUnits();

    // Resize the thread temporaries
    m_Count.SetSize(numberOfThreads);
    m_SumOfSquares.SetSize(numberOfThreads);
    m_ThreadSum.SetSize(numberOfThreads);
    m_ThreadMin.SetSize(numberOfThreads);
    m_ThreadMax.SetSize(numberOfThreads);

    // Initialize the temporaries
    m_Count.Fill(NumericTraits< SizeValueType >::Zero);
    m_ThreadSum.Fill(NumericTraits< RealType >::Zero);
    m_SumOfSquares.Fill(NumericTraits< RealType >::Zero);
    m_ThreadMin.Fill( NumericTraits< PixelType >::max() );
    m_ThreadMax.Fill( NumericTraits< PixelType >::NonpositiveMin() );
  }

  template< typename TInputImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::AfterThreadedGenerateData()
  {
    ThreadIdType    i;
    SizeValueType   count;
    RealType        sumOfSquares;

    ThreadIdType numberOfThreads = this->GetNumberOfWorkUnits();

    PixelType minimum;
    PixelType maximum;
    RealType  mean;
    RealType  sigma;
    RealType  variance;
    RealType  sum;

    sum = sumOfSquares = NumericTraits< RealType >::Zero;
    count = 0;

    // Find the min/max over all threads and accumulate count, sum and
    // sum of squares
    minimum = NumericTraits< PixelType >::max();
    maximum = NumericTraits< PixelType >::NonpositiveMin();
    for ( i = 0; i < numberOfThreads; i++ )
    {
      count += m_Count[i];
      sum += m_ThreadSum[i];
      sumOfSquares += m_SumOfSquares[i];

      if ( m_ThreadMin[i] < minimum )
      {
        minimum = m_ThreadMin[i];
      }
      if ( m_ThreadMax[i] > maximum )
      {
        maximum = m_ThreadMax[i];
      }
    }
    // compute statistics
    mean = sum / static_cast< RealType >( count );

    // unbiased estimate
    variance = ( sumOfSquares - ( sum * sum / static_cast< RealType >( count ) ) )
      / ( static_cast< RealType >( count ) - 1 );
    sigma = std::sqrt(variance);

    // Set the outputs
    this->GetMinimumOutput()->Set(minimum);
    this->GetMaximumOutput()->Set(maximum);
    this->GetMeanOutput()->Set(mean);
    this->GetSigmaOutput()->Set(sigma);
    this->GetVarianceOutput()->Set(variance);
    this->GetSumOutput()->Set(sum);
  }

  template< typename TInputImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TInputImage, TMaskImage >
    ::ThreadedGenerateData(const RegionType & outputRegionForThread,
    ThreadIdType threadId)
  {
    const SizeValueType size0 = outputRegionForThread.GetSize(0);
    if( size0 == 0)
    {
      return;
    }
    RealType  realValue;
    PixelType value;

    RealType sum = NumericTraits< RealType >::Zero;
    RealType sumOfSquares = NumericTraits< RealType >::Zero;
    SizeValueType count = NumericTraits< SizeValueType >::Zero;
    PixelType min = NumericTraits< PixelType >::max();
    PixelType max = NumericTraits< PixelType >::NonpositiveMin();

    ImageScanlineConstIterator< TInputImage > it (this->GetInput(),  outputRegionForThread);

    // support progress methods/callbacks
    const size_t numberOfLinesToProcess = outputRegionForThread.GetNumberOfPixels() / size0;
    ProgressReporter progress( this, threadId, numberOfLinesToProcess );

    // do the work
    while ( !it.IsAtEnd() )
    {
      while ( !it.IsAtEndOfLine() )
      {
        bool isValid = true;

        if(m_Mask.IsNotNull())
        {
          typename InputImageType::IndexType index = it.GetIndex();
          typename InputImageType::PointType point;
          this->GetInput()->TransformIndexToPhysicalPoint(index, point);
          if (this->m_Mask->TransformPhysicalPointToIndex(point, index))
          {
            isValid = this->m_Mask->GetPixel(index) > 0.0;
          };
        }

        if (isValid)
        {
          value = it.Get();
          realValue = static_cast< RealType >( value );
          if ( value < min )
          {
            min = value;
          }
          if ( value > max )
          {
            max  = value;
          }

          sum += realValue;
          sumOfSquares += ( realValue * realValue );
          ++count;
        }

        ++it;
      }
      it.NextLine();
      progress.CompletedPixel();
    }

    m_ThreadSum[threadId] = sum;
    m_SumOfSquares[threadId] = sumOfSquares;
    m_Count[threadId] = count;
    m_ThreadMin[threadId] = min;
    m_ThreadMax[threadId] = max;
  }

  template< typename TImage, typename TMaskImage >
  void
    MaskedStatisticsImageFilter< TImage, TMaskImage >
    ::PrintSelf(std::ostream & os, Indent indent) const
  {
    Superclass::PrintSelf(os, indent);

    os << indent << "Minimum: "
      << static_cast< typename NumericTraits< PixelType >::PrintType >( this->GetMinimum() ) << std::endl;
    os << indent << "Maximum: "
      << static_cast< typename NumericTraits< PixelType >::PrintType >( this->GetMaximum() ) << std::endl;
    os << indent << "Sum: "      << this->GetSum() << std::endl;
    os << indent << "Mean: "     << this->GetMean() << std::endl;
    os << indent << "Sigma: "    << this->GetSigma() << std::endl;
    os << indent << "Variance: " << this->GetVariance() << std::endl;
  }
} // end namespace itk
#endif
