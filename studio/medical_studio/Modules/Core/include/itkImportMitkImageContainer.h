/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef __itkImportMitkImageContainer_h
#define __itkImportMitkImageContainer_h

#include <itkImportImageContainer.h>
#include <mitkImageAccessorBase.h>
#include <mitkImageDataItem.h>

namespace itk
{
  /** \class ImportMitkImageContainer
   * Defines an itk::Image front-end to an mitk::Image. This container
   * conforms to the ImageContainerInterface. This is a full-fleged Object,
   * so there is modification time, debug, and reference count information.
   *
   * Template parameters for ImportMitkImageContainer:
   *
   * TElementIdentifier =
   *     An INTEGRAL type for use in indexing the imported buffer.
   *
   * TElement =
   *    The element type stored in the container.
   */

  template <typename TElementIdentifier, typename TElement>
  class ImportMitkImageContainer : public ImportImageContainer<TElementIdentifier, TElement>
  {
  public:
    /** Standard class typedefs. */
    typedef ImportMitkImageContainer Self;
    typedef Object Superclass;
    typedef SmartPointer<Self> Pointer;
    typedef SmartPointer<const Self> ConstPointer;

    /** Save the template parameters. */
    typedef TElementIdentifier ElementIdentifier;
    typedef TElement Element;

    /** Method for creation through the object factory. */
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      /** Standard part of every itk Object. */
      itkTypeMacro(ImportMitkImageContainer, ImportImageContainer);

    ///** Get the pointer from which the image data is imported. */
    // TElement *GetImportPointer() {return m_ImportPointer;};

    /** \brief Set the mitk::ImageDataItem to be imported  */
    // void SetImageDataItem(mitk::ImageDataItem* imageDataItem);
    void SetImageAccessor(mitk::ImageAccessorBase *imageAccess, size_t noBytes);

  protected:
    ImportMitkImageContainer();
    ~ImportMitkImageContainer() override;

    /** PrintSelf routine. Normally this is a protected internal method. It is
     * made public here so that Image can call this method.  Users should not
     * call this method but should call Print() instead. */
    void PrintSelf(std::ostream &os, Indent indent) const override;

  private:
    ImportMitkImageContainer(const Self &); // purposely not implemented
    void operator=(const Self &);           // purposely not implemented

    // mitk::ImageDataItem::Pointer m_ImageDataItem;
    mitk::ImageAccessorBase *m_imageAccess;
  };

} // end namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_ImportMitkImageContainer(_, EXPORT, x, y)                                                         \
  namespace itk                                                                                                        \
  {                                                                                                                    \
    _(2(class EXPORT ImportMitkImageContainer<ITK_TEMPLATE_2 x>))                                                      \
    namespace Templates                                                                                                \
    {                                                                                                                  \
      typedef ImportMitkImageContainer<ITK_TEMPLATE_2 x> ImportMitkImageContainer##y;                                  \
    }                                                                                                                  \
  }

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImportMitkImageContainer.txx"
#endif

#endif
