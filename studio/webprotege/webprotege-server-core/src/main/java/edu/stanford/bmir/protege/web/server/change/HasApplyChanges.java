package edu.stanford.bmir.protege.web.server.change;

import edu.stanford.bmir.protege.web.shared.permissions.PermissionDeniedException;
import edu.stanford.bmir.protege.web.shared.user.UserId;

/**
 * @author Matthew Horridge, Stanford University, Bio-Medical Informatics Research Group, Date: 25/03/2014
 */
public interface HasApplyChanges {
    <R> ChangeApplicationResult<R> applyChanges(UserId userId, ChangeListGenerator<R> changeListGenerator) throws PermissionDeniedException;
}
