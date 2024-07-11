/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef BERRYMULTISTATUS_H_
#define BERRYMULTISTATUS_H_

#include "berryStatus.h"

#include <org_blueberry_core_runtime_Export.h>

namespace berry {

/**
 * A concrete multi-status implementation,
 * suitable either for instantiating or subclassing.
 * <p>
 * This class can be used without OSGi running.
 * </p>
 */
class org_blueberry_core_runtime_EXPORT MultiStatus : public Status {

private:

  /** List of child statuses.
   */
  QList<IStatus::Pointer> children;


  Severity GetMaxSeverity(const QList<Pointer> &children) const;

public:

  berryObjectMacro(berry::MultiStatus);

  /**
   * Creates and returns a new multi-status object with the given children.
   *
   * @param pluginId the unique identifier of the relevant plug-in
   * @param code the plug-in-specific status code
   * @param newChildren the list of children status objects
   * @param message a human-readable message, localized to the
   *    current locale
   * @param sl
   */
  MultiStatus(const QString& pluginId, int code, const QList<IStatus::Pointer>& newChildren,
              const QString& message, const SourceLocation& sl);

  /**
   * Creates and returns a new multi-status object with the given children.
   *
   * @param pluginId the unique identifier of the relevant plug-in
   * @param code the plug-in-specific status code
   * @param newChildren the list of children status objects
   * @param message a human-readable message, localized to the
   *    current locale
   * @param exception a low-level exception.
   * @param sl
   */
  MultiStatus(const QString& pluginId, int code, const QList<IStatus::Pointer>& newChildren,
              const QString& message, const ctkException& exception, const SourceLocation& sl);

  /**
   * Creates and returns a new multi-status object with no children.
   *
   * @param pluginId the unique identifier of the relevant plug-in
   * @param code the plug-in-specific status code
   * @param message a human-readable message, localized to the
   *    current locale
   * @param sl
   */
  MultiStatus(const QString& pluginId, int code, const QString& message,
              const SourceLocation& sl);

  /**
   * Creates and returns a new multi-status object with no children.
   *
   * @param pluginId the unique identifier of the relevant plug-in
   * @param code the plug-in-specific status code
   * @param message a human-readable message, localized to the
   *    current locale
   * @param exception a low-level exception, or <code>null</code> if not
   *    applicable
   * @param sl
   */
  MultiStatus(const QString& pluginId, int code, const QString& message,
              const ctkException& exception, const SourceLocation& sl);

  /**
   * Adds the given status to this multi-status.
   *
   * @param status the new child status
   */
  void Add(IStatus::Pointer status);

  /**
   * Adds all of the children of the given status to this multi-status.
   * Does nothing if the given status has no children (which includes
   * the case where it is not a multi-status).
   *
   * @param status the status whose children are to be added to this one
   */
  void AddAll(IStatus::Pointer status);

  /* (Intentionally not javadoc'd)
   * Implements the corresponding method on <code>IStatus</code>.
   */
  QList<IStatus::Pointer> GetChildren() const override;

  /* (Intentionally not javadoc'd)
   * Implements the corresponding method on <code>IStatus</code>.
   */
  bool IsMultiStatus() const override;

  /**
   * Merges the given status into this multi-status.
   * Equivalent to <code>Add(status)</code> if the
   * given status is not a multi-status.
   * Equivalent to <code>AddAll(status)</code> if the
   * given status is a multi-status.
   *
   * @param status the status to merge into this one
   */
  void Merge(const IStatus::Pointer& status);

  /**
   * Returns a string representation of the status, suitable
   * for debugging purposes only.
   */
  QString ToString() const override;

};

}

#endif /* BERRYMULTISTATUS_H_ */
