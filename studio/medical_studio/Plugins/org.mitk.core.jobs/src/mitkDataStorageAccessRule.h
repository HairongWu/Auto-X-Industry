/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkDataStorageAccessRule_h
#define mitkDataStorageAccessRule_h

#include <org_mitk_core_jobs_Export.h>

#include "berryISchedulingRule.h"
#include "berryObject.h"
#include "mitkDataNode.h"
#include "mitkDataStorage.h"
#include "mitkStandaloneDataStorage.h"

namespace mitk {

 /**
  *@class DataStorageAccessRule
  *
  *@brief The DataStorageAccessRule inherits from the ISchedulingRule class. DataStorageAccessRule are used to restrict the adding and
  * removing of DataStorage nodes in multi-threaded scenarios. Only DataStorageNodes within different branches can be modified
  * concurrently. The idea and its restrictions is explained in the sections and diagrams below.
  *
  * <h2>the IsScheduling(...) method :</h2>
  * <p>
  * returns true or false depending if conflictions with another rule are found
  * </p>
  *
  * <h3>the rule behavior if jobs holing add rules of an DataTree node</h3>
  * <p>
  * two add rules are always allowed since there are no conflictions when adding nodes concurrently. The final order the nodes are finally added has
  * to be checked by the programmer of the particular job
  * \image html TwoAddRules.png
  * </p>
  *
  * <h3>the rule behavior when two jobs holding remove rules of a DataNode</h3>
  * <p>
  * two jobs holding remove rules can be executed concurrently since all removing scenarios do not cause conflictions. If two jobs are
  * trying to remove the same DataTree node the job by itself needs to check  if the node is still available before executing the removing
  * command
  * \image html TwoRemoveRules.png
  * </p>
  *
  * <h3>the rule behavior of a jobs that is holding an add rule compared with a job that is holding a remove rule on a
  * DataNode</h3>
  * <p>
  * adding and removing of DataTree nodes concurrently can cause serious errors and needs to be restricted. Performing add and remove
  * operations on different DataStorage branches can be done concurrently since no conflictions are expected.
  * the performing of add and remove operation on the same DataNode, its parent nodes or child nodes of the same branch
  * by different jobs is not allowed. Jobs holding rules that are trying to perform such operations are blocked until the running job is done.
  * \image html AddandRemoveRule.png
  * </p>
  *
  * <h2>the Contains method (...) method :</h2>
  * <p>
  * only necessary for a specific type of scheduling rules. Has to be used if IScheduling rules are composed into hierarchies.
  * In such scenarios the contains(...) method specifies the hierarchical relationships among the locks. For example if a method tries to acquire a specific     * rule to lock a specific directory it needs to check if no job is holding a rule for one or more subdirectories.  For all cases in which no composing of
  * IScheduling rules is needed the Contains(...) method only needs to check if two jobs are holding exactly the same IScheduling rule on the same object.
  * Normally this can be achieved by just calling the IsConflicting(...) method.
  * </p>
  *
  *@author Jan Woerner
  */




  class MITK_JOBS_EXPORT DataStorageAccessRule : public berry::ISchedulingRule {

  public:


  enum RuleType {ADD_RULE = 0, REMOVE_RULE} ;

  RuleType m_Rule;

  berryObjectMacro(DataStorageAccessRule);

  DataStorageAccessRule (mitk::DataStorage::Pointer myDataStorage, mitk::DataNode::Pointer myDataNode,
                         DataStorageAccessRule::RuleType myRule) ;

  bool Contains (berry::ISchedulingRule::Pointer otherISchedulingRule) const override;
  bool IsConflicting (berry::ISchedulingRule::Pointer otherISchedulingRule) const override;






private:


 /**
  * Returns false, identifying no conflictions between two DataStorageAccessRules.
  * Two add and remove rules do work together. From importance is that jobs using this rule need to check if the
  * node operating on is still available or already deleted by another job. The DataStorageAccessRule only checks if conflictions could
  * occur if the removing or adding of nodes is performed currently.
  */
  bool CompareTwoAddorRemoveRules() const ;
 /**
  * searches for conflictions of an add DataStorageAccessRule with a remove DataStorageAccess rule
  * if the add and remove rule do operate in different branches, no conflictions are expected and false is returned
  */
  bool CompareAddandRemoveRules(mitk::DataStorageAccessRule::Pointer sptr_otherDataStorageAccessRule) const;
  /**
   * for internal use only,
   * checks if the jobs that are to be compared do hold DataStorageAccessRule on the same
   * DataStorage. Jobs that do operate on different DataStorage do not conflict and false is returned
   */
  bool CompareDataStorages(mitk::DataStorage::Pointer otherDataStorage) const;
  /**
   * for internal use only
   * validates if the DataTree node of a particular DataStorageAccess rule belongs to the DataStorage specified within the particular rule.
   * if not the rule is invalid and false is returned. No conflictions can be expected
   */
  bool TestDataNode(mitk::DataNode::Pointer dataTreeToBeStored) const;

  /**
   * returns a pointer to the specified DataStorage node
   */
  mitk::DataNode::Pointer GetDataNode() const;

  /**
   * returns a pointer to the specified DataStorage
   */
  mitk::DataStorage::Pointer GetDataStorage() const;

  mitk::DataStorageAccessRule::RuleType GetRuleType() const;




  DataStorage::Pointer m_sptrMyDataStorage ;
  DataNode::Pointer m_sptrMyDataNode ;


  };

 }

#endif
