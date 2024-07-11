/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPropertyExtensions_h
#define mitkPropertyExtensions_h

#include <map>
#include <mitkIPropertyExtensions.h>

namespace mitk
{
  class PropertyExtensions : public IPropertyExtensions
  {
  public:
    PropertyExtensions();
    ~PropertyExtensions() override;

    bool AddExtension(const std::string &propertyName,
                      PropertyExtension::Pointer extension,
                      const std::string &className,
                      bool overwrite) override;
    PropertyExtension::Pointer GetExtension(const std::string &propertyName, const std::string &className) override;
    bool HasExtension(const std::string &propertyName, const std::string &className) override;
    void RemoveAllExtensions(const std::string &className) override;
    void RemoveExtension(const std::string &propertyName, const std::string &className) override;

  private:
    typedef std::map<std::string, PropertyExtension::Pointer> ExtensionMap;
    typedef ExtensionMap::const_iterator ExtensionMapConstIterator;
    typedef ExtensionMap::iterator ExtensionMapIterator;

    PropertyExtensions(const PropertyExtensions &);
    PropertyExtensions &operator=(const PropertyExtensions &);

    std::map<std::string, ExtensionMap> m_Extensions;
  };
}

#endif
