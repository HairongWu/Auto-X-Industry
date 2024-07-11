/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkSurfaceInterpolationController_h
#define mitkSurfaceInterpolationController_h

#include <mitkDataStorage.h>
#include <mitkLabelSetImage.h>
#include <mitkLabel.h>
#include <mitkSurface.h>

#include <MitkSurfaceInterpolationExports.h>

namespace mitk
{
  class ComputeContourSetNormalsFilter;
  class CreateDistanceImageFromSurfaceFilter;
  class LabelSetImage;
  class ReduceContourSetFilter;

  class MITKSURFACEINTERPOLATION_EXPORT SurfaceInterpolationController : public itk::Object
  {
  public:
    mitkClassMacroItkParent(SurfaceInterpolationController, itk::Object);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    struct MITKSURFACEINTERPOLATION_EXPORT ContourPositionInformation
    {
      Surface::ConstPointer Contour;
      PlaneGeometry::ConstPointer Plane;
      Label::PixelType LabelValue;
      TimeStepType TimeStep;

      ContourPositionInformation()
        : Plane(nullptr),
          LabelValue(std::numeric_limits<Label::PixelType>::max()),
          TimeStep(std::numeric_limits<TimeStepType>::max())
      {
      }

      ContourPositionInformation(Surface::ConstPointer contour,
        PlaneGeometry::ConstPointer plane,
        Label::PixelType labelValue,
        TimeStepType timeStep)
        :
        Contour(contour),
        Plane(plane),
        LabelValue(labelValue),
        TimeStep(timeStep)
      {
      }

      bool IsPlaceHolder() const
      {
        return Contour.IsNull();
      }
    };

    typedef std::vector<ContourPositionInformation> CPIVector;

    static SurfaceInterpolationController *GetInstance();

    /**
     * @brief Adds new extracted contours to the list. If one or more contours at a given position
     *        already exist they will be updated respectively
     */
    void AddNewContours(const std::vector<ContourPositionInformation>& newCPIs, bool reinitializeAction = false, bool silent = false);

    /**
     * @brief Removes the contour for a given plane for the current selected segmentation
     * @param contourInfo the contour which should be removed
     * @param keepPlaceholderForUndo
     * @return true if a contour was found and removed, false if no contour was found
     */
    bool RemoveContour(ContourPositionInformation contourInfo, bool keepPlaceholderForUndo = false);

    void RemoveObservers();

    /**
     * @brief Performs the interpolation.
     *
     */
    void Interpolate(const LabelSetImage* segmentationImage, LabelSetImage::LabelValueType labelValue, TimeStepType timeStep);

    /**
     * @brief Get the Result of the interpolation operation.
     *
     * @return mitk::Surface::Pointer
     */
    mitk::Surface::Pointer GetInterpolationResult(const LabelSetImage* segmentationImage, LabelSetImage::LabelValueType labelValue, TimeStepType timeStep);

    /**
     * @brief Sets the minimum spacing of the current selected segmentation
     * This is needed since the contour points we reduced before they are used to interpolate the surface.
     *
     * @param minSpacing Parameter to set
     * @param minSpacing Parameter to set
     */
    void SetMinSpacing(double minSpacing);

    /**
     * @brief Sets the minimum spacing of the current selected segmentation
     * This is needed since the contour points we reduced before they are used to interpolate the surface
     * @param maxSpacing Set the max Spacing for interpolation
     */
    void SetMaxSpacing(double maxSpacing);

    /**
     * Sets the volume i.e. the number of pixels that the distance image should have
     * By evaluation we found out that 50.000 pixel delivers a good result
     */
    void SetDistanceImageVolume(unsigned int distImageVolume);

    /**
     * @brief Get the current selected segmentation for which the interpolation is performed
     * @return the current segmentation image
     */
    mitk::LabelSetImage* GetCurrentSegmentation();

    void SetDataStorage(DataStorage::Pointer ds);

    /**
     * Sets the current list of contourpoints which is used for the surface interpolation
     * @param currentSegmentationImage The current selected segmentation
     */
    void SetCurrentInterpolationSession(LabelSetImage* currentSegmentationImage);

    /**
     * @brief Remove interpolation session
     * @param segmentationImage the session to be removed
     */
    void RemoveInterpolationSession(const LabelSetImage* segmentationImage);

    /**
     * @brief Removes all sessions
     */
    void RemoveAllInterpolationSessions();

    /**
     * @brief Get the Contours at a certain timeStep and layerID.
     *
     * @param timeStep Time Step from which to get the contours.
     * @param labelValue label from which to get the contours.
     * @return std::vector<ContourPositionInformation> Returns contours.
     */
    CPIVector* GetContours(LabelSetImage::LabelValueType labelValue, TimeStepType timeStep);

    std::vector<LabelSetImage::LabelValueType> GetAffectedLabels(const LabelSetImage* seg, TimeStepType timeStep, const PlaneGeometry* plane) const;

    /**
     * @brief Triggered with the "Reinit Interpolation" action. The contours are used to repopulate the
     * @brief Triggered with the "Reinit Interpolation" action. The contours are used to repopulate the
     *        surfaceInterpolator data structures so that interpolation can be performed after reloading data.
     */
    void CompleteReinitialization(const std::vector<ContourPositionInformation>& newCPIs);

    /**
     * @brief Removes contours of a particular label and at a given time step for the current session/segmentation.
     *
     * @param segmentationImage
     * @param label Label of contour to remove.
     * @param timeStep Time step in which to remove the contours.
     * @remark if the label or time step does not exist, nothing happens.
     */
    void RemoveContours(const LabelSetImage* segmentationImage, mitk::Label::PixelType label, TimeStepType timeStep);

    /**
     * @brief Removes contours of a particular label and at a given time step for the current session/segmentation.
     *
     * @param segmentationImage
     * @param label Label of contour to remove.
     * @remark if the label or time step does not exist, nothing happens.
     */
    void RemoveContours(const LabelSetImage* segmentationImage, mitk::Label::PixelType label);

    unsigned int GetNumberOfInterpolationSessions();

    /**
     * @brief Get the Segmentation Image Node object
     *
     * @return DataNode* returns the DataNode containing the segmentation image.
     */
    mitk::DataNode* GetSegmentationImageNode() const;

  protected:
    SurfaceInterpolationController();

    ~SurfaceInterpolationController() override;

    template <typename TPixel, unsigned int VImageDimension>
    void GetImageBase(itk::Image<TPixel, VImageDimension> *input, itk::ImageBase<3>::Pointer &result);

  private:

    /**
     * @brief
     *
     * @param caller
     * @param event
     */
    void OnSegmentationDeleted(const itk::Object *caller, const itk::EventObject &event);

    /**
     * @brief Function that removes contours of a particular label when the "Remove Label" event is triggered in the labelSetImage.
     * @brief Function that removes contours of a particular label when the "Remove Label" event is triggered in the labelSetImage.
     *
     */
    void OnRemoveLabel(const itk::Object* caller, const itk::EventObject& event);

    /**
     * @brief When a new contour is added to the pipeline or an existing contour is replaced,
     *        the plane geometry information of that contour is added as a child node to the
     *        current node of the segmentation image. This is useful in the retrieval of contour information
     *        when data is reloaded after saving.
     *
     * @param contourInfo contourInfo struct to add to data storage.
     */
    void AddPlaneGeometryNodeToDataStorage(const ContourPositionInformation& contourInfo) const;

    DataStorage::SetOfObjects::ConstPointer GetPlaneGeometryNodeFromDataStorage(const DataNode* segNode) const;
    DataStorage::SetOfObjects::ConstPointer GetPlaneGeometryNodeFromDataStorage(const DataNode* segNode, LabelSetImage::LabelValueType labelValue) const;
    DataStorage::SetOfObjects::ConstPointer GetPlaneGeometryNodeFromDataStorage(const DataNode* segNode, LabelSetImage::LabelValueType labelValue, TimeStepType timeStep) const;

    /**
     * Adds Contours from the active Label to the interpolation pipeline
     */
    void AddActiveLabelContoursForInterpolation(ReduceContourSetFilter* reduceFilter, const LabelSetImage* segmentationImage, LabelSetImage::LabelValueType labelValue, TimeStepType timeStep);

    /**
     * @brief Clears the interpolation data structures. Called from CompleteReinitialization().
     *
     */
    void ClearInterpolationSession();

    void RemoveObserversInternal(const mitk::LabelSetImage* segmentationImage);

    /**
     * @brief Add contour to the interpolation pipeline
     *
     * @param contourInfo Contour information to be added
     * @param reinitializationAction If the contour is coming from a reinitialization process or not
     */
    void AddToCPIMap(ContourPositionInformation& contourInfo, bool reinitializationAction = false);

    unsigned int m_DistanceImageVolume;
    mitk::DataStorage::Pointer m_DataStorage;

    WeakPointer<LabelSetImage> m_SelectedSegmentation;
  };
}

#endif
