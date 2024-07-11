/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkImageStatisticsCalculator_h
#define mitkImageStatisticsCalculator_h

#include <MitkImageStatisticsExports.h>
#include <mitkImage.h>
#include <mitkMaskGenerator.h>
#include <mitkImageStatisticsContainer.h>

namespace mitk
{
    class MITKIMAGESTATISTICS_EXPORT ImageStatisticsCalculator: public itk::Object
    {
    public:
        /** Standard Self typedef */
        typedef ImageStatisticsCalculator        Self;
        typedef itk::Object                         Superclass;
        typedef itk::SmartPointer< Self >           Pointer;
        typedef itk::SmartPointer< const Self >     ConstPointer;

        /** Method for creation through the object factory. */
        itkNewMacro(Self); /** Runtime information support. */
        itkTypeMacro(ImageStatisticsCalculator, itk::Object);

        typedef double statisticsValueType;
        typedef std::map<std::string, statisticsValueType> statisticsMapType;
        typedef itk::Statistics::Histogram<double> HistogramType;
        typedef unsigned short MaskPixelType;
        using LabelIndex = ImageStatisticsContainer::LabelValueType;

        /**Documentation
        @brief Set the image for which the statistics are to be computed.*/
        void SetInputImage(const mitk::Image* image);

        /**Documentation
        @brief Set the mask generator that creates the mask which is to be used to calculate statistics. If no more mask is desired simply set @param mask to nullptr*/
        void SetMask(mitk::MaskGenerator* mask);

        /**Documentation
        @brief Set this if more than one mask should be applied (for instance if a IgnorePixelValueMask were to be used alongside with a segmentation).
        Both masks are combined using pixel wise AND operation. The secondary mask does not have to be the same size than the primary but they need to have some overlap*/
        void SetSecondaryMask(mitk::MaskGenerator* mask);

        /**Documentation
        @brief Set number of bins to be used for histogram statistics. If Bin size is set after number of bins, bin size will be used instead!*/
        void SetNBinsForHistogramStatistics(unsigned int nBins);

        /**Documentation
        @brief Retrieve the number of bins used for histogram statistics. Careful: The return value does not indicate whether NBins or BinSize is used.
        That solely depends on which parameter has been set last.*/
        unsigned int GetNBinsForHistogramStatistics() const;

        /**Documentation
        @brief Set bin size to be used for histogram statistics. If nbins is set after bin size, nbins will be used instead!*/
        void SetBinSizeForHistogramStatistics(double binSize);

        /**Documentation
        @brief Retrieve the bin size for histogram statistics. Careful: The return value does not indicate whether NBins or BinSize is used.
        That solely depends on which parameter has been set last.*/
        double GetBinSizeForHistogramStatistics() const;

        /**Documentation
        @brief Returns the statistics. If these requested statistics are not computed yet the computation is done as well.
         */
        ImageStatisticsContainer* GetStatistics();

    protected:
        ImageStatisticsCalculator(){
            m_nBinsForHistogramStatistics = 100;
            m_binSizeForHistogramStatistics = 10;
            m_UseBinSizeOverNBins = false;
        };


    private:
        //Calculates statistics for each timestep for image
        template < typename TPixel, unsigned int VImageDimension >
        void InternalCalculateStatisticsUnmasked(const itk::Image< TPixel, VImageDimension >* image, TimeStepType timeStep);

        template < typename TPixel, unsigned int VImageDimension >
        void InternalCalculateStatisticsMasked(const itk::Image< TPixel, VImageDimension >* image, TimeStepType timeStep);

        template < typename TPixel, unsigned int VImageDimension >
        double GetVoxelVolume(const itk::Image<TPixel, VImageDimension>* image) const;

        bool IsUpdateRequired() const;

        mitk::Image::ConstPointer m_Image;
        mitk::Image::ConstPointer m_ImageTimeSlice;
        mitk::Image::ConstPointer m_InternalImageForStatistics;

        mitk::MaskGenerator::Pointer m_MaskGenerator;
        mitk::Image::ConstPointer m_InternalMask;

        mitk::MaskGenerator::Pointer m_SecondaryMaskGenerator;
        mitk::Image::ConstPointer m_SecondaryMask;

        unsigned int m_nBinsForHistogramStatistics;
        double m_binSizeForHistogramStatistics;
        bool m_UseBinSizeOverNBins;

        ImageStatisticsContainer::Pointer m_StatisticContainer;
    };

}
#endif
