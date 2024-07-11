/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkImageDataItem_h
#define mitkImageDataItem_h

#include "mitkCommon.h"
#include <MitkCoreExports.h>
#include "mitkImageDescriptor.h"

class vtkImageData;

namespace mitk
{
  class PixelType;
  class ImageVtkReadAccessor;
  class ImageVtkWriteAccessor;

  class Image;

  //##Documentation
  //## @brief Internal class for managing references on sub-images
  //##
  //## ImageDataItem is a container for image data which is used internal in
  //## mitk::Image to handle the communication between the different data types for images
  //## used in MITK (mitk::Image, vtkImageData). Common for these image data
  //## types is the actual image data, but they differ in representation of pixel type etc.
  //##
  //## The class is mainly used to extract sub-images inside of mitk::Image, like single slices etc.
  //## It should not be used outside of this.
  //##
  //## @param manageMemory Determines if image data is removed while destruction of ImageDataItem or not.
  //## @ingroup Data
  class MITKCORE_EXPORT ImageDataItem : public itk::LightObject
  {
    friend class ImageAccessorBase;
    friend class ImageWriteAccessor;
    friend class ImageReadAccessor;

    template <class TPixel, unsigned int VDimension>
    friend class ImagePixelAccessor;

    friend class Image;

    //  template<class TOutputImage>
    //  friend class ImageToItk;

  public:
    typedef itk::SmartPointer<mitk::Image> ImagePointer;
    typedef itk::SmartPointer<const mitk::Image> ImageConstPointer;

    mitkClassMacroItkParent(ImageDataItem, itk::LightObject);

    itkCloneMacro(ImageDataItem);
    itk::LightObject::Pointer InternalClone() const override;

    ImageDataItem(const ImageDataItem &aParent,
                  const mitk::ImageDescriptor::Pointer desc,
                  int timestep,
                  unsigned int dimension,
                  void *data = nullptr,
                  bool manageMemory = false,
                  size_t offset = 0);

    ~ImageDataItem() override;

    ImageDataItem(const mitk::ImageDescriptor::Pointer desc, int timestep, void *data, bool manageMemory);

    ImageDataItem(const mitk::PixelType &type,
                  int timestep,
                  unsigned int dimension,
                  unsigned int *dimensions,
                  void *data,
                  bool manageMemory);

    ImageDataItem(const ImageDataItem &other);

    bool IsComplete() const { return m_IsComplete; }
    void SetComplete(bool complete) { m_IsComplete = complete; }
    int GetOffset() const { return m_Offset; }
    PixelType GetPixelType() const { return *m_PixelType; }
    void SetTimestep(int t) { m_Timestep = t; }
    void SetManageMemory(bool b) { m_ManageMemory = b; }
    int GetDimension() const { return m_Dimension; }
    int GetDimension(int i) const
    {
      int returnValue = 0;

      // return the true size if dimension available
      if (i < (int)m_Dimension)
        returnValue = m_Dimensions[i];

      return returnValue;
    }

    ImageDataItem::ConstPointer GetParent() const { return m_Parent; }
    /**
     * @brief GetVtkImageAccessor Returns a vtkImageDataItem, if none is present, a new one is constructed by the
     * ConstructVtkImageData method.
     *                            Due to historical development of MITK and VTK, the vtkImage origin is explicitly set
     * to
     * (0, 0, 0) for 3D images.
     *                            See bug 5050 for detailed information.
     * @return Pointer of type ImageVtkReadAccessor
     */
    ImageVtkReadAccessor *GetVtkImageAccessor(ImageConstPointer) const;
    ImageVtkWriteAccessor *GetVtkImageAccessor(ImagePointer);

    // Returns if image data should be deleted on destruction of ImageDataItem.
    bool GetManageMemory() const { return m_ManageMemory; }
    virtual void ConstructVtkImageData(ImageConstPointer) const;

    size_t GetSize() const { return m_Size; }
    virtual void Modified() const;

  protected:

    /**Helper function to allow friend classes to access m_Data without changing their code.
    * Moved to protected visibility because only friends are allowed to access m_Data directly.
    * Other classes should used ImageWriteAccessor::GetData() or ImageReadAccessor::GetData()
    * to get access.*/
    void* GetData() const { return m_Data; }

    unsigned char *m_Data;

    PixelType *m_PixelType;

    bool m_ManageMemory;

    mutable vtkImageData *m_VtkImageData;
    mutable ImageVtkReadAccessor *m_VtkImageReadAccessor;
    ImageVtkWriteAccessor *m_VtkImageWriteAccessor;
    int m_Offset;

    bool m_IsComplete;

    size_t m_Size;

  private:
    void ComputeItemSize(const unsigned int *dimensions, unsigned int dimension);

    ImageDataItem::ConstPointer m_Parent;

    unsigned int m_Dimension;

    unsigned int m_Dimensions[MAX_IMAGE_DIMENSIONS];

    int m_Timestep;
  };

} // namespace mitk

#endif
