/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkImageStatisticsContainer_h
#define mitkImageStatisticsContainer_h

#include <MitkImageStatisticsExports.h>
#include <mitkBaseData.h>
#include <itkHistogram.h>
#include <mitkLabelSetImage.h>
#include <mitkImageStatisticsConstants.h>

#include <boost/variant.hpp>

namespace mitk
{

  /**
   @brief Container class for storing a StatisticsObject for each time step.

   Stored statistics are:
   - for the defined statistics, see GetAllStatisticNames
   - Histogram of Pixel Values
  */
  class MITKIMAGESTATISTICS_EXPORT ImageStatisticsContainer : public mitk::BaseData
  {
  public:
    mitkClassMacro(ImageStatisticsContainer, mitk::BaseData);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    using HistogramType = itk::Statistics::Histogram<double>;
    using LabelValueType = LabelSetImage::LabelValueType;

    void SetRequestedRegionToLargestPossibleRegion() override {}

    bool RequestedRegionIsOutsideOfTheBufferedRegion() override { return false; }

    bool VerifyRequestedRegion() override { return true; }

    void SetRequestedRegion(const itk::DataObject*) override {}

    /**
    @brief Container class for storing the computed image statistics.
    @details The statistics are stored in a map <name,value> with value as boost::variant<RealType, VoxelCountType, IndexType >.
    The type used to create the boost::variant is important as only this type can be recovered later on.
    */
    class MITKIMAGESTATISTICS_EXPORT ImageStatisticsObject {
    public:
      ImageStatisticsObject();

      using RealType = double;
      using IndexType = vnl_vector<int>;
      using VoxelCountType = unsigned long;

      using StatisticsVariantType = boost::variant<RealType, VoxelCountType, IndexType >;

      /**
      @brief Adds a statistic to the statistics object
      @details if already a statistic with that name is included, it is overwritten
      */
      void AddStatistic(const std::string_view key, StatisticsVariantType value);

      using StatisticNameVector = std::vector<std::string>;

      /**
      @brief Returns the names of the default statistics
      @details The order is derived from the image statistics plugin.
      */
      static const StatisticNameVector& GetDefaultStatisticNames();

      /**
      @brief Returns the names of all custom statistics (defined at runtime and no default names).
      */
      const StatisticNameVector& GetCustomStatisticNames() const;

      /**
      @brief Returns the names of all statistics (default and custom defined)
      Additional custom keys are added at the end in a sorted order.
      */
      StatisticNameVector GetAllStatisticNames() const;

      StatisticNameVector GetExistingStatisticNames() const;

      bool HasStatistic(const std::string_view name) const;

      /**
      @brief Converts the requested value to the defined type
      @param name defined string on creation (AddStatistic)
      @exception if no statistics with key name was found.
      */
      template <typename TType>
      TType GetValueConverted(const std::string_view name) const
      {
        auto value = GetValueNonConverted(name);
        return boost::get<TType>(value);
      }

      /**
      @brief Returns the requested value
      @exception if no statistics with key name was found.
      */
      StatisticsVariantType GetValueNonConverted(const std::string_view name) const;

      void Reset();

      HistogramType::ConstPointer m_Histogram=nullptr;
    private:

      using StatisticsMapType = std::map < std::string, StatisticsVariantType, std::less<>>;

      StatisticsMapType m_Statistics;
      StatisticNameVector m_CustomNames;
      static const StatisticNameVector m_DefaultNames;
    };

    using StatisticsVariantType = ImageStatisticsObject::StatisticsVariantType;
    using RealType = ImageStatisticsObject::RealType;
    using IndexType = ImageStatisticsObject::IndexType;
    using VoxelCountType = ImageStatisticsObject::VoxelCountType;

    using TimeStepVectorType = std::vector<TimeStepType>;
    TimeStepVectorType GetExistingTimeSteps(LabelValueType labelValue) const;

    /** Value that can be used to query for the statistic if no mask was provided.*/
    static constexpr LabelValueType NO_MASK_LABEL_VALUE = Label::UNLABELED_VALUE;
    using LabelValueVectorType = LabelSetImage::LabelValueVectorType;
    LabelValueVectorType GetExistingLabelValues() const;

    /**
    @brief Deletes all stored values*/
    void Reset();

    const ImageStatisticsObject& GetStatistics(LabelValueType labelValue, TimeStepType timeStep) const;

    /**
    @brief Sets the statisticObject for the given Timestep
    @pre timeStep must be valid
    */
    void SetStatistics(LabelValueType labelValue, TimeStepType timeStep, const ImageStatisticsObject& statistics);

    /**
    @brief Checks if the Time step exists
    @pre timeStep must be valid
    */
    bool StatisticsExist(LabelValueType labelValue, TimeStepType timeStep) const;

    /**
    /brief Returns the histogram of the passed time step.
    @pre timeStep must be valid*/
    const HistogramType* GetHistogram(LabelValueType labelValue, TimeStepType timeStep) const;

    bool IgnoresZeroVoxel() const;

    bool IsWIP() const;

  protected:
    ImageStatisticsContainer();
    void PrintSelf(std::ostream &os, itk::Indent indent) const override;

  private:
    itk::LightObject::Pointer InternalClone() const override;

    using TimeStepMapType = std::map<TimeStepType, ImageStatisticsObject>;
    using LabelMapType = std::map<LabelValueType, TimeStepMapType>;

    LabelMapType m_LabelTimeStep2StatisticsMap;
  };

  MITKIMAGESTATISTICS_EXPORT ImageStatisticsContainer::ImageStatisticsObject::StatisticNameVector GetAllStatisticNames(const ImageStatisticsContainer* container);
  MITKIMAGESTATISTICS_EXPORT ImageStatisticsContainer::ImageStatisticsObject::StatisticNameVector GetAllStatisticNames(std::vector<ImageStatisticsContainer::ConstPointer> containers);
}
#endif
