/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPropertyAliases_h
#define mitkPropertyAliases_h

#include <map>
#include <mitkIPropertyAliases.h>

namespace mitk
{
  class PropertyAliases : public IPropertyAliases
  {
  public:
    PropertyAliases();
    ~PropertyAliases() override;

    bool AddAlias(const std::string &propertyName, const std::string &alias, const std::string &className) override;
    std::vector<std::string> GetAliases(const std::string &propertyName, const std::string &className) override;
    std::string GetPropertyName(const std::string &alias, const std::string &className) override;
    bool HasAliases(const std::string &propertyName, const std::string &className) override;
    void RemoveAlias(const std::string &propertyName, const std::string &alias, const std::string &className) override;
    void RemoveAliases(const std::string &propertyName, const std::string &className) override;
    void RemoveAllAliases(const std::string &className) override;

  private:
    typedef std::map<std::string, std::vector<std::string>> AliasesMap;
    typedef AliasesMap::const_iterator AliasesMapConstIterator;
    typedef AliasesMap::iterator AliasesMapIterator;

    PropertyAliases(const PropertyAliases &);
    PropertyAliases &operator=(const PropertyAliases &);

    std::map<std::string, AliasesMap> m_Aliases;
  };
}

#endif
