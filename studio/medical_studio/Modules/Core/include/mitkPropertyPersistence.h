/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPropertyPersistence_h
#define mitkPropertyPersistence_h

#include <map>
#include <mitkIPropertyPersistence.h>

namespace mitk
{
  class PropertyPersistence : public IPropertyPersistence
  {
  public:
    PropertyPersistence();
    ~PropertyPersistence() override;

    typedef IPropertyPersistence::InfoResultType InfoResultType;

    bool AddInfo(const PropertyPersistenceInfo *info, bool overwrite) override;
    InfoResultType GetInfo(const std::string &propertyName, bool allowNameRegEx) const override;
    InfoResultType GetInfo(const std::string &propertyName,
                           const MimeTypeNameType &mime,
                           bool allowMimeWildCard,
                           bool allowNameRegEx) const override;
    InfoResultType GetInfoByKey(const std::string &persistenceKey, bool allowKeyRegEx) const override;
    bool HasInfo(const std::string &propertyName, bool allowNameRegEx) const override;
    void RemoveAllInfo() override;
    void RemoveInfo(const std::string &propertyName) override;
    void RemoveInfo(const std::string &propertyName, const MimeTypeNameType &mime) override;

  private:
    typedef std::multimap<const std::string, PropertyPersistenceInfo::ConstPointer> InfoMap;

    /**Helper function that selects */
    using SelectFunctionType = std::function<bool(const InfoMap::value_type &)>;
    static InfoMap SelectInfo(const InfoMap &infoMap, const SelectFunctionType &selectFunction);

    PropertyPersistence(const PropertyPersistence &);
    PropertyPersistence &operator=(const PropertyPersistence &);

    InfoMap m_InfoMap;
  };

  /**Creates an unmanaged (!) instance of PropertyPersistence for testing purposes.*/
  MITKCORE_EXPORT IPropertyPersistence *CreateTestInstancePropertyPersistence();
}

#endif
