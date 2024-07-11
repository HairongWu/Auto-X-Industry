/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkNodePredicateDataType_h
#define mitkNodePredicateDataType_h

#include "mitkDataNode.h"
#include "mitkNodePredicateBase.h"
#include <string>

namespace mitk
{
  //##Documentation
  //## @brief Predicate that evaluates if the given DataNodes data object is of a specific data type
  //##
  //## The data type must be specified in the constructor as a string. The string must equal the result
  //## value of the requested data types GetNameOfClass() method.
  //##
  //## @ingroup DataStorage
  class MITKCORE_EXPORT NodePredicateDataType : public NodePredicateBase
  {
  public:
    mitkClassMacro(NodePredicateDataType, NodePredicateBase);
    mitkNewMacro1Param(NodePredicateDataType, const char *);

    //##Documentation
    //## @brief Standard Destructor
    ~NodePredicateDataType() override;

    //##Documentation
    //## @brief Checks, if the nodes data object is of a specific data type
    bool CheckNode(const mitk::DataNode *node) const override;

  protected:
    //##Documentation
    //## @brief Protected constructor, use static instantiation functions instead
    NodePredicateDataType(const char *datatype);

    std::string m_ValidDataType;
  };

  /**
   * \brief Tests for type compatibility (dynamic_cast).
   *
   * In contrast to NodePredicateDataType this class also accepts derived types.
   * E.g. if you query for type BaseData, you will also get Image and Surface objects.
   *
   * The desired type is given as a template parameter, the constructor takes no other parameters.
   */
  template <class T>
  class TNodePredicateDataType : public NodePredicateBase
  {
  public:
    mitkClassMacro(TNodePredicateDataType, NodePredicateBase);
    itkFactorylessNewMacro(TNodePredicateDataType);

    ~TNodePredicateDataType() override {}
    //##Documentation
    //## @brief Checks, if the nodes data object is of a specific data type (casts)
    bool CheckNode(const mitk::DataNode *node) const override
    {
      return node && node->GetData() && dynamic_cast<T *>(node->GetData());
    }

  protected:
    //##Documentation
    //## @brief Protected constructor, use static instantiation functions instead
    TNodePredicateDataType() {}
  };

} // namespace mitk

#endif
