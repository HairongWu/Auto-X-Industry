/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkNodePredicateFunction_h
#define mitkNodePredicateFunction_h

#include <mitkNodePredicateBase.h>
#include <MitkCoreExports.h>
#include <functional>

namespace mitk
{
  /**
  * \brief Predicate that transfers the check to a specific Callable.
  *
  * Callables include free functions, bound member functions, function
  * objects, and lambda expressions. The signature of the Callable must
  * match NodePredicateFunction::CheckNode.
  *
  * \ingroup DataStorage
  */
  class MITKCORE_EXPORT NodePredicateFunction : public NodePredicateBase
  {
  public:
    using FunctionType = std::function<bool (const mitk::DataNode *)>;

    mitkClassMacro(NodePredicateFunction, NodePredicateBase);
    mitkNewMacro1Param(NodePredicateFunction, const FunctionType&);

    ~NodePredicateFunction() override;

    bool CheckNode(const mitk::DataNode *node) const override;

  protected:
    explicit NodePredicateFunction(const FunctionType &function);

    FunctionType m_Function;
  };
}

#endif
