/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkIDICOMTagsOfInterest_h
#define mitkIDICOMTagsOfInterest_h

#include <mitkServiceInterface.h>
#include <string>
#include <vector>
#include <mitkDICOMTagsOfInterestHelper.h>
#include <MitkDICOMExports.h>

namespace mitk
{
  /**
   * \ingroup MicroServices_Interfaces
   * \brief Interface of DICOM tags of interest service.
   *
   * This service allows you to manage the tags of interest (toi).
   * All registered toi will be extracted when loading dicom data and stored as properties in the corresponding
   * base data object. In addition the service can (if available) use IPropertyPersistance and IPropertyAliases
   * to ensure that the tags of interests are also persisted and have a human readable alias.
   */
  class MITKDICOM_EXPORT IDICOMTagsOfInterest
  {
  public:
    virtual ~IDICOMTagsOfInterest();

    /** \brief Add an tag to the TOI.
      * If the tag was already added it will be overwritten with the passed values.
      * \param[in] tag Tag that should be added.
      * \param[in] makePersistant Indicates if the tag should be made persistent if possible via the IPropertyPersistence service.
      */
    virtual void AddTagOfInterest(const DICOMTagPath& tag, bool makePersistant = true) = 0;

    /** Returns the map of all tags of interest. Key is the property name. Value is the DICOM tag.*/
    virtual DICOMTagPathMapType GetTagsOfInterest() const = 0;

    /** Indicates if the given tag is already a tag of interest.*/
    virtual bool HasTag(const DICOMTagPath& tag) const = 0;

    /** \brief Remove specific tag. If it not exists the function will do nothing.
      * \param[in] tag Tag that should be removed.
      */
    virtual void RemoveTag(const DICOMTagPath& tag) = 0;

    /** \brief Remove all tags.
      */
    virtual void RemoveAllTags() = 0;
  };
}

MITK_DECLARE_SERVICE_INTERFACE(mitk::IDICOMTagsOfInterest, "org.mitk.IDICOMTagsOfInterest")

#endif
