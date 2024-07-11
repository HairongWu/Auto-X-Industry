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
#ifndef __itkEnhancedHistogramToRunLengthFeaturesFilter_h
#define __itkEnhancedHistogramToRunLengthFeaturesFilter_h

#include "itkHistogram.h"
#include "itkMacro.h"
#include "itkProcessObject.h"
#include "itkSimpleDataObjectDecorator.h"

namespace itk {
  namespace Statistics {
    /** \class EnhancedHistogramToRunLengthFeaturesFilter
    *  \brief This class computes texture feature coefficients from a grey level
    * run-length matrix.
    *
    * By default, run length features are computed for each spatial
    * direction and then averaged afterward, so it is possible to access the
    * standard deviations of the texture features. These values give a clue as
    * to texture anisotropy. However, doing this is much more work, because it
    * involved computing one for each offset given. To compute a single matrix
    * using the first offset, call FastCalculationsOn(). If this is called,
    * then the texture standard deviations will not be computed (and will be set
    * to zero), but texture computation will be much faster.
    *
    * This class is templated over the input histogram type.
    *
    * Print references:
    * M. M. Galloway. Texture analysis using gray level run lengths. Computer
    * Graphics and Image Processing, 4:172-179, 1975.
    *
    * A. Chu, C. M. Sehgal, and J. F. Greenleaf. Use of gray value distribution of
    * run lengths for texture analysis.  Pattern Recognition Letters, 11:415-420,
    * 1990.
    *
    * B. R. Dasarathy and E. B. Holder. Image characterizations based on joint
    * gray-level run-length distributions. Pattern Recognition Letters, 12:490-502,
    * 1991.
    *
    * IJ article: https://hdl.handle.net/1926/1374
    *
    * \sa ScalarImageToRunLengthFeaturesFilter
    * \sa ScalarImageToRunLengthMatrixFilter
    * \sa EnhancedHistogramToRunLengthFeaturesFilter
    *
    * \author: Nick Tustison
    * \ingroup ITKStatistics
    */

    template< typename THistogram >
    class EnhancedHistogramToRunLengthFeaturesFilter : public ProcessObject
    {
    public:
      /** Standard typedefs */
      typedef EnhancedHistogramToRunLengthFeaturesFilter     Self;
      typedef ProcessObject                          Superclass;
      typedef SmartPointer<Self>                     Pointer;
      typedef SmartPointer<const Self>               ConstPointer;

      /** Run-time type information (and related methods). */
      itkTypeMacro( EnhancedHistogramToRunLengthFeaturesFilter, ProcessObject );

      /** standard New() method support */
      itkNewMacro( Self );

      typedef THistogram                                      HistogramType;
      typedef typename HistogramType::Pointer                 HistogramPointer;
      typedef typename HistogramType::ConstPointer            HistogramConstPointer;
      typedef typename HistogramType::MeasurementType         MeasurementType;
      typedef typename HistogramType::MeasurementVectorType   MeasurementVectorType;
      typedef typename HistogramType::IndexType               IndexType;
      typedef typename HistogramType::
        TotalAbsoluteFrequencyType                            FrequencyType;

      /** Method to Set/Get the input Histogram */
      using Superclass::SetInput;
      void SetInput ( const HistogramType * histogram );
      const HistogramType * GetInput() const;

      /** Smart Pointer type to a DataObject. */
      typedef DataObject::Pointer                   DataObjectPointer;

      /** Type of DataObjects used for scalar outputs */
      typedef SimpleDataObjectDecorator<MeasurementType>     MeasurementObjectType;

      /** Methods to return the short run emphasis. */
      MeasurementType GetShortRunEmphasis() const;
      const MeasurementObjectType* GetShortRunEmphasisOutput() const;

      /** Methods to return the long run emphasis. */
      MeasurementType GetLongRunEmphasis() const;
      const MeasurementObjectType* GetLongRunEmphasisOutput() const;

      /** Methods to return the grey level nonuniformity. */
      MeasurementType GetGreyLevelNonuniformity() const;
      const MeasurementObjectType* GetGreyLevelNonuniformityOutput() const;

      /** Methods to return the grey level nonuniformity. */
      MeasurementType GetGreyLevelNonuniformityNormalized() const;
      const MeasurementObjectType* GetGreyLevelNonuniformityNormalizedOutput() const;

      /** Methods to return the run length nonuniformity. */
      MeasurementType GetRunLengthNonuniformity() const;
      const MeasurementObjectType* GetRunLengthNonuniformityOutput() const;

      /** Methods to return the run length nonuniformity. */
      MeasurementType GetRunLengthNonuniformityNormalized() const;
      const MeasurementObjectType* GetRunLengthNonuniformityNormalizedOutput() const;

      /** Methods to return the low grey level run emphasis. */
      MeasurementType GetLowGreyLevelRunEmphasis() const;
      const MeasurementObjectType* GetLowGreyLevelRunEmphasisOutput() const;

      /** Methods to return the high grey level run emphasis. */
      MeasurementType GetHighGreyLevelRunEmphasis() const;
      const MeasurementObjectType* GetHighGreyLevelRunEmphasisOutput() const;

      /** Methods to return the short run low grey level run emphasis. */
      MeasurementType GetShortRunLowGreyLevelEmphasis() const;
      const MeasurementObjectType* GetShortRunLowGreyLevelEmphasisOutput() const;

      /** Methods to return the short run high grey level run emphasis. */
      MeasurementType GetShortRunHighGreyLevelEmphasis() const;
      const MeasurementObjectType* GetShortRunHighGreyLevelEmphasisOutput() const;

      /** Methods to return the long run low grey level run emphasis. */
      MeasurementType GetLongRunLowGreyLevelEmphasis() const;
      const MeasurementObjectType* GetLongRunLowGreyLevelEmphasisOutput() const;

      /** Methods to return the long run high grey level run emphasis. */
      MeasurementType GetLongRunHighGreyLevelEmphasis() const;
      const MeasurementObjectType* GetLongRunHighGreyLevelEmphasisOutput() const;

      /** Methods to return the long run high grey level run emphasis. */
      MeasurementType GetRunPercentage() const;
      const MeasurementObjectType* GetRunPercentageOutput() const;

      /** Methods to return the long run high grey level run emphasis. */
      MeasurementType GetNumberOfRuns() const;
      const MeasurementObjectType* GetNumberOfRunsOutput() const;

      /** Methods to return the grey level variance. */
      MeasurementType GetGreyLevelVariance() const;
      const MeasurementObjectType* GetGreyLevelVarianceOutput() const;

      /** Methods to return the run length variance. */
      MeasurementType GetRunLengthVariance() const;
      const MeasurementObjectType* GetRunLengthVarianceOutput() const;

      /** Methods to return the run entropy. */
      MeasurementType GetRunEntropy() const;
      const MeasurementObjectType* GetRunEntropyOutput() const;

      itkGetMacro( TotalNumberOfRuns, unsigned long );

      itkGetConstMacro(NumberOfVoxels, unsigned long);
      itkSetMacro(NumberOfVoxels, unsigned long);

      /** Run-length feature types */
      typedef enum
      {
        ShortRunEmphasis,
        LongRunEmphasis,
        GreyLevelNonuniformity,
        GreyLevelNonuniformityNormalized,
        RunLengthNonuniformity,
        RunLengthNonuniformityNormalized,
        LowGreyLevelRunEmphasis,
        HighGreyLevelRunEmphasis,
        ShortRunLowGreyLevelEmphasis,
        ShortRunHighGreyLevelEmphasis,
        LongRunLowGreyLevelEmphasis,
        LongRunHighGreyLevelEmphasis,
        RunPercentage,
        NumberOfRuns,
        GreyLevelVariance,
        RunLengthVariance,
        RunEntropy
      }  RunLengthFeatureName;

      /** convenience method to access the run length values */
      MeasurementType GetFeature( RunLengthFeatureName name );

    protected:
      EnhancedHistogramToRunLengthFeaturesFilter();
      ~EnhancedHistogramToRunLengthFeaturesFilter() override {};
      void PrintSelf(std::ostream& os, Indent indent) const ITK_OVERRIDE;

      /** Make a DataObject to be used for output output. */
      typedef ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
      using Superclass::MakeOutput;
      DataObjectPointer MakeOutput( DataObjectPointerArraySizeType ) ITK_OVERRIDE;

      void GenerateData() ITK_OVERRIDE;

    private:
      EnhancedHistogramToRunLengthFeaturesFilter(const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented

      unsigned long                           m_TotalNumberOfRuns;
      unsigned long                           m_NumberOfVoxels;
    };
  } // end of namespace Statistics
} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkEnhancedHistogramToRunLengthFeaturesFilter.hxx"
#endif

#endif
