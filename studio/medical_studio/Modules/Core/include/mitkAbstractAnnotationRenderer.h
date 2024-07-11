/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkAbstractAnnotationRenderer_h
#define mitkAbstractAnnotationRenderer_h

#include "mitkAnnotation.h"
#include "mitkServiceInterface.h"
#include "usGetModuleContext.h"
#include "usServiceTracker.h"
#include <MitkCoreExports.h>
#include <mitkCommon.h>

namespace mitk
{
  class BaseRenderer;

  /** @brief Baseclass of Annotation layouters
   * An AbstractAnnotationRenderer can be implemented to control a set of Annotation by means of position and size.
   * @ingroup Annotation
  */
  class MITKCORE_EXPORT AbstractAnnotationRenderer : public us::ServiceTracker<Annotation>
  {
  public:
    typedef us::ServiceTracker<Annotation> Superclass;
    AbstractAnnotationRenderer(const std::string &rendererID, const std::string &arID);

    /** \brief virtual destructor in order to derive from this class */
    ~AbstractAnnotationRenderer() override;

    const std::string GetID() const;
    const std::string GetRendererID() const;

    void CurrentBaseRendererChanged();

    virtual void OnRenderWindowModified() {}
    void RemoveAllAnnotation();

    void Update();
    static const std::string US_INTERFACE_NAME;
    static const std::string US_PROPKEY_ID;
    static const std::string US_PROPKEY_RENDERER_ID;

  protected:
    BaseRenderer *GetCurrentBaseRenderer();

  private:
    /** \brief copy constructor */
    AbstractAnnotationRenderer(const AbstractAnnotationRenderer &);

    /** \brief assignment operator */
    AbstractAnnotationRenderer &operator=(const AbstractAnnotationRenderer &);

    TrackedType AddingService(const ServiceReferenceType &reference) override;

    void ModifiedService(const ServiceReferenceType & /*reference*/, TrackedType tracked) override;

    void RemovedService(const ServiceReferenceType & /*reference*/, TrackedType tracked) override;

    virtual void OnAnnotationRenderersChanged() {}
    const std::string m_RendererID;
    const std::string m_ID;
  };

} // namespace mitk

MITK_DECLARE_SERVICE_INTERFACE(mitk::AbstractAnnotationRenderer, "org.mitk.services.AbstractAnnotationRenderer")

#endif
