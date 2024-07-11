/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef _BERRY_IPROGRESSMONITORWITHBLOCKING_H
#define _BERRY_IPROGRESSMONITORWITHBLOCKING_H

#include "berryObject.h"
#include <org_blueberry_core_jobs_Export.h>
#include "berryIStatus.h"
#include "berryIProgressMonitor.h"

namespace berry
{

/**
 * An extension to the IProgressMonitor interface for monitors that want to
 * support feedback when an activity is blocked due to concurrent activity in
 * another thread.
 * <p>
 * When a monitor that supports this extension is passed to an operation, the
 * operation should call <code>setBlocked</code> whenever it knows that it
 * must wait for a lock that is currently held by another thread. The operation
 * should continue to check for and respond to cancellation requests while
 * blocked. When the operation is no longer blocked, it must call <code>clearBlocked</code>
 * to clear the blocked state.
 * <p>
 * This interface can be used without OSGi running.
 * </p><p>
 * Clients may implement this interface.
 * </p>
 * @see IProgressMonitor
 */
struct BERRY_JOBS IProgressMonitorWithBlocking: public IProgressMonitor
{

  berryObjectMacro(berry::IProgressMonitorWithBlocking);

  /**
   * Indicates that this operation is blocked by some background activity. If
   * a running operation ever calls <code>setBlocked</code>, it must
   * eventually call <code>clearBlocked</code> before the operation
   * completes.
   * <p>
   * If the caller is blocked by a currently executing job, this method will return
   * an <code>IJobStatus</code> indicating the job that is currently blocking
   * the caller. If this blocking job is not known, this method will return a plain
   * informational <code>IStatus</code> object.
   * </p>
   *
   * @param reason an optional status object whose message describes the
   * reason why this operation is blocked, or <code>null</code> if this
   * information is not available.
   * @see #ClearBlocked
   */
   virtual void SetBlocked(IStatus::Pointer reason)= 0;

  /**
   * Clears the blocked state of the running operation. If a running
   * operation ever calls <code>setBlocked</code>, it must eventually call
   * <code>clearBlocked</code> before the operation completes.
   *
   * @see #SetBlocked
   */
   virtual void  ClearBlocked() = 0;


};

}

#endif /* _BERRY_IPROGRESSMONITORWITHBLOCKING_H */
