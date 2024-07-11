/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPlanarFigureMaskGenerator_h
#define mitkPlanarFigureMaskGenerator_h

#include <MitkImageStatisticsExports.h>
#include <itkImage.h>
#include <mitkImage.h>
#include <mitkMaskGenerator.h>
#include <mitkPlanarFigure.h>
#include <vtkSmartPointer.h>

namespace mitk
{
  /**
   * \class PlanarFigureMaskGenerator
   * \brief Derived from MaskGenerator. This class is used to convert a mitk::PlanarFigure into a binary image mask
   */
  class MITKIMAGESTATISTICS_EXPORT PlanarFigureMaskGenerator : public MaskGenerator
  {
  public:
    /** Standard Self typedef */
    typedef PlanarFigureMaskGenerator Self;
    typedef MaskGenerator Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self); /** Runtime information support. */
      itkTypeMacro(PlanarFigureMaskGenerator, MaskGenerator);

    unsigned int GetNumberOfMasks() const override;

    void SetPlanarFigure(mitk::PlanarFigure* planarFigure);

    mitk::Image::ConstPointer GetReferenceImage() override;

    itkGetConstMacro(PlanarFigureAxis, unsigned int);
    itkGetConstMacro(PlanarFigureSlice, unsigned int);

    /** Helper function that indicates if a passed planar geometry is tilted regarding a given geometry and its main axis.
     *@pre If either planarGeometry or geometry is nullptr it will return false.*/
    static bool CheckPlanarFigureIsNotTilted(const PlaneGeometry* planarGeometry, const BaseGeometry *geometry);

  protected:
    PlanarFigureMaskGenerator()
      : Superclass(),
        m_ReferenceImage(nullptr),
        m_PlanarFigureAxis(0),
        m_InternalMaskUpdateTime(0),
        m_PlanarFigureSlice(0)
    {
      m_InternalMask = mitk::Image::New();
    }

    Image::ConstPointer DoGetMask(unsigned int) override;

  private:
    void CalculateMask();

    template <typename TPixel, unsigned int VImageDimension>
    void InternalCalculateMaskFromClosedPlanarFigure(const itk::Image<TPixel, VImageDimension> *image, unsigned int axis);

    template <typename TPixel, unsigned int VImageDimension>
    void InternalCalculateMaskFromOpenPlanarFigure(const itk::Image<TPixel, VImageDimension> *image, unsigned int axis);

    mitk::Image::ConstPointer Extract2DImageSlice(const Image* input, unsigned int axis, unsigned int slice) const;

    /** Helper function that deduces if the passed vector is equal to one of the primary axis of the geometry.*/
    static bool GetPrincipalAxis(const BaseGeometry *geometry, Vector3D vector, unsigned int &axis);

    /** Connection from ITK to VTK */
    template <typename ITK_Exporter, typename VTK_Importer>
    void ConnectPipelines(ITK_Exporter exporter, vtkSmartPointer<VTK_Importer> importer)
    {
      importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());

      importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
      importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
      importer->SetSpacingCallback(exporter->GetSpacingCallback());
      importer->SetOriginCallback(exporter->GetOriginCallback());
      importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());

      importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());

      importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
      importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
      importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
      importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
      importer->SetCallbackUserData(exporter->GetCallbackUserData());
    }

    /** Connection from VTK to ITK */
    template <typename VTK_Exporter, typename ITK_Importer>
    void ConnectPipelines(vtkSmartPointer<VTK_Exporter> exporter, ITK_Importer importer)
    {
      importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());

      importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
      importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
      importer->SetSpacingCallback(exporter->GetSpacingCallback());
      importer->SetOriginCallback(exporter->GetOriginCallback());
      importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());

      importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());

      importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
      importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
      importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
      importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
      importer->SetCallbackUserData(exporter->GetCallbackUserData());
    }

    bool IsUpdateRequired() const;

    mitk::PlanarFigure::Pointer m_PlanarFigure;
    itk::Image<unsigned short, 2>::Pointer m_InternalITKImageMask2D;
    mitk::Image::ConstPointer m_ReferenceImage;
    unsigned int m_PlanarFigureAxis;
    unsigned long m_InternalMaskUpdateTime;
    unsigned int m_PlanarFigureSlice;
    mitk::Image::Pointer m_InternalMask;
  };

} // namespace mitk

#endif
