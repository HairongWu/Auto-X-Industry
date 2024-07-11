/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkMultiLabelEvents_h
#define mitkMultiLabelEvents_h

#include <itkEventObject.h>
#include <mitkLabel.h>

#include <MitkMultilabelExports.h>

namespace mitk
{
#define mitkMultiLabelEventMacroDeclaration(classname, super, IDType) \
  class MITKMULTILABEL_EXPORT classname : public super     \
  {                                                        \
  public:                                                  \
    using Self = classname;                                \
    using Superclass = super;                              \
    classname() = default;                                 \
    classname(IDType value);                               \
    classname(const Self & s);                             \
    virtual ~classname() override;                         \
    virtual const char *                                   \
    GetEventName() const override;                         \
    virtual bool                                           \
    CheckEvent(const itk::EventObject * e) const override; \
    virtual itk::EventObject *                             \
    MakeObject() const override;                           \
                                                           \
  private:                                                 \
    void                                                   \
    operator=(const Self &);                               \
  };                                                       \
  static_assert(true, "Compile time eliminated. Used to require a semi-colon at end of macro.")

#define mitkMultiLabelEventMacroDefinition(classname, super, IDType)         \
  classname::classname(const classname & s)                                  \
    : super(s){};                                                            \
  classname::classname(IDType value): super(value) {}                        \
  classname::~classname() {}                                                 \
  const char * classname::GetEventName() const { return #classname; }        \
  bool         classname::CheckEvent(const itk::EventObject * e) const       \
  {                                                                          \
    if (!super::CheckEvent(e)) return false;                                 \
    return (dynamic_cast<const classname *>(e) != nullptr);                  \
  }                                                                          \
  itk::EventObject * classname::MakeObject() const { return new classname; } \
  static_assert(true, "Compile time eliminated. Used to require a semi-colon at end of macro.")

  /** Base event class for all events that are about a label in a MultiLabel class.
  *
  * It has a member that indicates the label id the event is refering to.
  * Use the ANY_LABEL value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every label and not just to a special one.
  */
  class MITKMULTILABEL_EXPORT AnyLabelEvent : public itk::ModifiedEvent
  {
  public:
    using Self = AnyLabelEvent;
    using Superclass = itk::ModifiedEvent;
    const static mitk::Label::PixelType ANY_LABEL = std::numeric_limits<mitk::Label::PixelType>::max();

    AnyLabelEvent() = default;
    AnyLabelEvent(Label::PixelType labelValue);
    AnyLabelEvent(const Self & s);
    ~AnyLabelEvent() override;
    const char * GetEventName() const override;
    bool CheckEvent(const itk::EventObject * e) const override;
    itk::EventObject * MakeObject() const override;

    void SetLabelValue(Label::PixelType labelValue);
    Label::PixelType GetLabelValue() const;
  private:
    void operator=(const Self &);
    Label::PixelType m_LabelValue = std::numeric_limits<mitk::Label::PixelType>::max();
  };

  /** Event class that is used to indicated if a label is added in a MultiLabel class.
  *
  * It has a member that indicates the label id the event is refering to.
  * Use the ANY_LABEL value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every label and not just to a special one.
  */
  mitkMultiLabelEventMacroDeclaration(LabelAddedEvent, AnyLabelEvent, Label::PixelType);

  /** Event class that is used to indicated if a label is modified in a MultiLabel class.
  *
  * It has a member that indicates the label id the event is refering to.
  * Use the ANY_LABEL value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every label and not just to a special one.
  */
  mitkMultiLabelEventMacroDeclaration(LabelModifiedEvent, AnyLabelEvent, Label::PixelType);

  /** Event class that is used to indicated if a label is removed in a MultiLabel class.
  *
  * It has a member that indicates the label id the event is refering to.
  * Use the ANY_LABEL value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every label and not just to a special one.
  */
  mitkMultiLabelEventMacroDeclaration(LabelRemovedEvent, AnyLabelEvent, Label::PixelType);

  /** Event class that is used to indicated if a set of labels is changed in a MultiLabel class.
  *
  * In difference to the other label events LabelsChangedEvent is send only *one time* after
  * the modification of the MultiLableImage instance is finished. So e.g. even if 4 labels are
  * changed by a merge operation, this event will only be sent once (compared to LabelRemoved
  * or LabelModified).
  * It has a member that indicates the label ids the event is refering to.
  */
  class MITKMULTILABEL_EXPORT LabelsChangedEvent : public itk::ModifiedEvent
  {
  public:
    using Self = LabelsChangedEvent;
    using Superclass = itk::ModifiedEvent;

    LabelsChangedEvent() = default;
    LabelsChangedEvent(std::vector<Label::PixelType> labelValues);
    LabelsChangedEvent(const Self& s);
    ~LabelsChangedEvent() override;
    const char* GetEventName() const override;
    bool CheckEvent(const itk::EventObject* e) const override;
    itk::EventObject* MakeObject() const override;

    void SetLabelValues(std::vector<Label::PixelType> labelValues);
    std::vector<Label::PixelType> GetLabelValues() const;
  private:
    void operator=(const Self&);
    std::vector<Label::PixelType> m_LabelValues;
  };

  /** Base event class for all events that are about a group in a MultiLabel class.
  *
  * It has a member that indicates the group id the event is refering to.
  * Use the ANY_GROUP value if you want to define an event (e.g. for adding an observer)
  * that reacts to every group and not just to a special one.
  */
  class MITKMULTILABEL_EXPORT AnyGroupEvent : public itk::ModifiedEvent
  {
  public:
    using GroupIndexType = std::size_t;
    using Self = AnyGroupEvent;
    using Superclass = itk::ModifiedEvent;
    const static GroupIndexType ANY_GROUP = std::numeric_limits<GroupIndexType>::max();

    AnyGroupEvent() = default;
    AnyGroupEvent(GroupIndexType groupID);
    AnyGroupEvent(const Self& s);
    ~AnyGroupEvent() override;
    const char* GetEventName() const override;
    bool CheckEvent(const itk::EventObject* e) const override;
    itk::EventObject* MakeObject() const override;

    void SetGroupID(GroupIndexType groupID);
    GroupIndexType GetGroupID() const;
  private:
    void operator=(const Self&);
    GroupIndexType m_GroupID = std::numeric_limits<GroupIndexType>::max();
  };

  /** Event class that is used to indicated if a group is added in a MultiLabel class.
  *
  * It has a member that indicates the group id the event is refering to.
  * Use the ANY_GROUP value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every group and not just to a special one.
  */
  mitkMultiLabelEventMacroDeclaration(GroupAddedEvent, AnyGroupEvent, AnyGroupEvent::GroupIndexType);

  /** Event class that is used to indicated if a group is modified in a MultiLabel class.
  *
  * It has a member that indicates the group id the event is refering to.
  * Use the ANY_GROUP value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every group and not just to a special one.
  */
  mitkMultiLabelEventMacroDeclaration(GroupModifiedEvent, AnyGroupEvent, AnyGroupEvent::GroupIndexType);

  /** Event class that is used to indicated if a group is removed in a MultiLabel class.
  *
  * It has a member that indicates the group id the event is refering to.
  * Use the ANY_GROUP value if you want to define an rvent (e.g. for adding an observer)
  * that reacts to every group and not just to a special one.
  */
  mitkMultiLabelEventMacroDeclaration(GroupRemovedEvent, AnyGroupEvent, AnyGroupEvent::GroupIndexType);

}

#endif
