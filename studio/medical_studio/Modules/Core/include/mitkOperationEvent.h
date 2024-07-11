/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkOperationEvent_h
#define mitkOperationEvent_h

#include "mitkOperation.h"
#include "mitkOperationActor.h"
#include "mitkUndoModel.h"
#include <MitkCoreExports.h>
#include <list>
#include <string>

namespace mitk
{
  //##Documentation
  //## @brief Represents an entry of the undo or redo stack.
  //##
  //## This basic entry includes a textual description of the item and a pair of IDs. Static
  //## member functions handle creation and incrementing of these IDs.
  //##
  //## The GroupEventID is intended for logical grouping of several related Operations.
  //## Currently this is used only by PointSetDataInteractor. How this is done and when to use
  //## GroupEventIDs is still undocumented.
  //## @ingroup Undo
  class MITKCORE_EXPORT UndoStackItem
  {
  public:
    UndoStackItem(std::string description = "");

    virtual ~UndoStackItem();

    //##Documentation
    //## @brief For combining operations in groups
    //##
    //## This ID is used in the undo mechanism.
    //## For separation of the separate operations
    //## If the GroupEventId of two OperationEvents is equal,
    //## then they share one group and will be undone in case of Undo(fine==false)
    static int GetCurrGroupEventId();

    //##Documentation
    //## @brief For combining operations in Objects
    //##
    //## This ID is used in the Undo-Mechanism.
    //## For separation of the separate operations
    //## If the ObjectEventId of two OperationEvents is equal,
    //## then they share one Object and will be undone in all cases of Undo(true and false).
    //## they shall not be separated, because they were produced to realize one object-change.
    //## for example: OE_statechange and OE_addlastpoint
    static int GetCurrObjectEventId();

    //##Documentation
    //## @brief Returns the GroupEventId for this object
    int GetGroupEventId();

    //##Documentation
    //## @brief Returns the ObjectEventId for this object
    int GetObjectEventId();

    //##Documentation
    //## @brief Returns the textual description of this object
    std::string GetDescription();

    virtual void ReverseOperations();
    virtual void ReverseAndExecute();

    //##Documentation
    //## @brief Increases the current ObjectEventId
    //## For example if a button click generates operations the ObjectEventId has to be incremented to be able to undo
    //the
    // operations.
    //## Difference between ObjectEventId and GroupEventId: The ObjectEventId capsulates all operations caused by one
    // event.
    //## A GroupEventId capsulates several ObjectEventIds so that several operations caused by several events can be
    // undone with one Undo call.
    static void IncCurrObjectEventId();

    //##Documentation
    //## @brief Increases the current GroupEventId
    //## For example if a button click generates operations the GroupEventId has to be incremented to be able to undo
    //the
    // operations.
    //## Difference between ObjectEventId and GroupEventId: The ObjectEventId capsulates all operations caused by one
    // event.
    //## A GroupEventId capsulates several ObjectEventIds so that several operations caused by several events can be
    // undone with one Undo call.
    static void IncCurrGroupEventId();

  protected:
    //##Documentation
    //## @brief true, if operation and undooperation have been swapped/changed
    bool m_Reversed;

  private:
    static int m_CurrObjectEventId;

    static int m_CurrGroupEventId;

    int m_ObjectEventId;

    int m_GroupEventId;

    std::string m_Description;

    UndoStackItem(UndoStackItem &);        // hide copy constructor
    void operator=(const UndoStackItem &); // hide operator=
  };

  //##Documentation
  //## @brief Represents a pair of operations: undo and the according redo.
  //##
  //## Additionally to the base class UndoStackItem, which only provides a description of an
  //## item, OperationEvent does the actual accounting of the undo/redo stack. This class
  //## holds two Operation objects (operation and its inverse operation) and the corresponding
  //## OperationActor. The operations may be swapped by the
  //## undo models, when an OperationEvent is moved from their undo to their redo
  //## stack or vice versa.
  //##
  //## Note, that memory management of operation and undooperation is done by this class.
  //## Memory of both objects is freed in the destructor. For this, the method IsValid() is needed which holds
  //## information of the state of m_Destination. In case the object referenced by m_Destination is already deleted,
  //## isValid() returns false.
  //## In more detail if the destination happens to be an itk::Object (often the case), OperationEvent is informed as
  //soon
  //## as the object is deleted - from this moment on the OperationEvent gets invalid. You should
  //## check this flag before you call anything on destination
  //##
  //## @ingroup Undo
  class MITKCORE_EXPORT OperationEvent : public UndoStackItem
  {
  public:
    //## @brief default constructor
    OperationEvent(OperationActor *destination,
                   Operation *operation,
                   Operation *undoOperation,
                   std::string description = "");

    //## @brief default destructor
    //##
    //## removes observers if destination is valid
    //## and frees memory referenced by m_Operation and m_UndoOperation
    ~OperationEvent() override;

    //## @brief Returns the operation
    Operation *GetOperation();

    //## @brief Returns the destination of the operations
    OperationActor *GetDestination();

    friend class UndoModel;

    //## @brief Swaps the two operations and sets a flag,
    //## that it has been swapped and doOp is undoOp and undoOp is doOp
    void ReverseOperations() override;

    //##reverses and executes both operations (used, when moved from undo to redo stack)
    void ReverseAndExecute() override;

    //## @brief returns true if the destination still is present
    //## and false if it already has been deleted
    virtual bool IsValid();

  protected:
    void OnObjectDeleted();

  private:
    // Has to be observed for itk::DeleteEvents.
    // When destination is deleted, this stack item is invalid!
    OperationActor *m_Destination;

    //## reference to the operation
    Operation *m_Operation;

    //## reference to the undo operation
    Operation *m_UndoOperation;

    //## hide copy constructor
    OperationEvent(OperationEvent &);
    //## hide operator=
    void operator=(const OperationEvent &);

    // observertag used to listen to m_Destination
    unsigned long m_DeleteTag;

    //## stores if destination is valid or already has been freed
    bool m_Invalid;
  };

} // namespace mitk

#endif
