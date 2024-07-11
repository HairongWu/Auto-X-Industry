/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkNodePredicateData_h
#define mitkNodePredicateData_h

#include "mitkNodePredicateBase.h"

namespace mitk
{
  class BaseData;

  //##Documentation
  //## @brief Predicate that evaluates if the given DataNodes data object pointer equals a given pointer
  //##
  //## NodePredicateData(nullptr) returns true if a DataNode does not have a data object (e.g. ->GetData() returns nullptr).
  //## This could return an unexpected number of nodes (e.g. the root node of the tree)
  //## @warning NodePredicateData holds a weak pointer to a BaseData! NodePredicateData p(mitk::BaseData::New()); will
  //not
  // work.
  //##          Intended use is: NodePredicateData p(myDataObject); result = myDataStorage->GetSubset(p); Then work with
  // result, do not reuse p later.
  //##
  //## @ingroup DataStorage
  class MITKCORE_EXPORT NodePredicateData : public NodePredicateBase
  {
  public:
    mitkClassMacro(NodePredicateData, NodePredicateBase);
    mitkNewMacro1Param(NodePredicateData, mitk::BaseData *);

    //##Documentation
    //## @brief Standard Destructor
    ~NodePredicateData() override;

    //##Documentation
    //## @brief Checks, if the nodes data object is of a specific data type
    bool CheckNode(const mitk::DataNode *node) const override;

  protected:
    //##Documentation
    //## @brief Protected constructor, use static instantiation functions instead
    NodePredicateData(mitk::BaseData *d);

    mitk::BaseData *m_DataObject;
  };
} // namespace mitk

#endif
