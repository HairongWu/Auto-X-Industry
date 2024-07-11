/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkGenericLookupTable_h
#define mitkGenericLookupTable_h

#include <map>
#include <sstream>
#include <cstdlib>
#include <string>

#include <itkDataObject.h>

#include "mitkNumericTypes.h"
#include <MitkCoreExports.h>

#include <nlohmann/json.hpp>

namespace mitk
{
  template <typename T> class GenericLookupTable;
  template <typename T> void from_json(const nlohmann::json&, GenericLookupTable<T>&);

  /**
   * @brief Template class for generating lookup-tables
   *
   * This class template can be instantiated for all classes/internal types that fulfills
   *  these requirements:
   * - an operator<< so that the properties value can be put into a std::stringstream
   * - an operator== so that two properties can be checked for equality
   *
   * The main purpose of this class is to be used in conjunction with
   * GenericLookupTableProperty. This enables passing of arbitrary lookup
   * tables to mappers to configure the rendering process.
   */
  template <typename T>
  class GenericLookupTable
  {
  public:
    typedef unsigned int IdentifierType;
    typedef T ValueType;
    typedef std::map<IdentifierType, ValueType> LookupTableType;

    typedef GenericLookupTable Self;

    GenericLookupTable() {}
    virtual ~GenericLookupTable() {}
    virtual const char *GetNameOfClass() const { return "GenericLookupTable"; }
    void SetTableValue(IdentifierType id, ValueType value) { m_LookupTable[id] = value; }
    bool ValueExists(IdentifierType id) const
    {
      auto it = m_LookupTable.find(id);
      return (it != m_LookupTable.end());
    }

    ValueType GetTableValue(IdentifierType id) const
    {
      auto it = m_LookupTable.find(id);
      if (it != m_LookupTable.end())
        return it->second;
      else
        throw std::range_error("id does not exist in the lookup table");
    }

    const LookupTableType &GetLookupTable() const { return m_LookupTable; }
    bool operator==(const Self &lookupTable) const { return (m_LookupTable == lookupTable.m_LookupTable); }
    bool operator!=(const Self &lookupTable) const { return !(m_LookupTable == lookupTable.m_LookupTable); }
    virtual Self &operator=(const Self &other) // \TODO: this needs to be unit tested!
    {
      if (this == &other)
      {
        return *this;
      }
      else
      {
        m_LookupTable.clear();
        m_LookupTable = other.m_LookupTable;
        return *this;
      }
    }

    friend void from_json<>(const nlohmann::json&, GenericLookupTable<T>&);

  protected:
    LookupTableType m_LookupTable;
  };

  template <typename T>
  void to_json(nlohmann::json& j, const GenericLookupTable<T>& t)
  {
    j = t.GetLookupTable();
  }

  template <typename T>
  void from_json(const nlohmann::json& j, GenericLookupTable<T>& t)
  {
    j.get_to(t.m_LookupTable);
  }

} // namespace mitk

/**
* Generates a specialized subclass of mitk::GenericLookupTable.
* This way, GetNameOfClass() returns the value provided by LookupTableName.
* Please see mitkProperties.h for examples.
* @param LookupTableName the name of the instantiation of GenericLookupTable
* @param Type the value type of the GenericLookupTable
*/
#define mitkSpecializeGenericLookupTable(LookupTableName, Type)                                                        \
                                                                                                                       \
  class MITKCORE_EXPORT LookupTableName : public GenericLookupTable<Type>                                              \
                                                                                                                       \
  {                                                                                                                    \
  public:                                                                                                              \
    typedef LookupTableName Self;                                                                                      \
    typedef GenericLookupTable<Type> Superclass;                                                                       \
    virtual const char *GetNameOfClass() const { return #LookupTableName; }                                            \
    LookupTableName() {}                                                                                               \
    virtual Superclass &operator=(const Superclass &other) { return Superclass::operator=(other); }                    \
    virtual ~LookupTableName() {}                                                                                      \
  };                                                                                                                   \
                                                                                                                       \
  MITKCORE_EXPORT std::ostream &operator<<(std::ostream &stream, const LookupTableName & /*l*/);

/**
* Generates the ostream << operator for the lookuptable. This definition
* of a global function must be in a cpp file, therefore it is split from the
* class declaration macro mitkSpecializeGenericLookupTable.
*/
#define mitkSpecializeGenericLookupTableOperator(LookupTableName)                                                      \
                                                                                                                       \
  std::ostream &mitk::operator<<(std::ostream &stream, const LookupTableName &l)                                       \
                                                                                                                       \
  {                                                                                                                    \
    typedef LookupTableName::LookupTableType::const_iterator IterType;                                                 \
    IterType e = l.GetLookupTable().end();                                                                             \
    IterType b = l.GetLookupTable().begin();                                                                           \
    stream << "[";                                                                                                     \
    for (IterType i = b; i != e; ++i)                                                                                  \
    {                                                                                                                  \
      if (i != b)                                                                                                      \
      {                                                                                                                \
        stream << ", ";                                                                                                \
      }                                                                                                                \
      stream << i->first << " -> " << i->second;                                                                       \
    }                                                                                                                  \
    return stream << "]";                                                                                              \
  };
#endif
