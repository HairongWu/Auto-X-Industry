package edu.stanford.bmir.protege.web.server.dispatch.handlers;

import edu.stanford.bmir.protege.web.server.dispatch.ApplicationActionHandler;
import edu.stanford.bmir.protege.web.server.dispatch.ExecutionContext;
import edu.stanford.bmir.protege.web.server.dispatch.RequestContext;
import edu.stanford.bmir.protege.web.server.dispatch.RequestValidator;
import edu.stanford.bmir.protege.web.server.dispatch.validators.UserIsProjectOwnerValidator;
import edu.stanford.bmir.protege.web.server.project.ProjectDetailsManager;
import edu.stanford.bmir.protege.web.shared.event.EventList;
import edu.stanford.bmir.protege.web.shared.event.EventTag;
import edu.stanford.bmir.protege.web.shared.event.ProjectMovedFromTrashEvent;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.shared.project.RemoveProjectFromTrashAction;
import edu.stanford.bmir.protege.web.shared.project.RemoveProjectsFromTrashResult;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.List;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 19/04/2013
 */
public class RemoveProjectsFromTrashActionHandler implements ApplicationActionHandler<RemoveProjectFromTrashAction, RemoveProjectsFromTrashResult> {

    private final ProjectDetailsManager projectDetailsManager;

    @Inject
    public RemoveProjectsFromTrashActionHandler(ProjectDetailsManager projectDetailsManager) {
        this.projectDetailsManager = projectDetailsManager;
    }

    @Nonnull
    @Override
    public Class<RemoveProjectFromTrashAction> getActionClass() {
        return RemoveProjectFromTrashAction.class;
    }

    @Nonnull
    @Override
    public RequestValidator getRequestValidator(@Nonnull RemoveProjectFromTrashAction action, @Nonnull RequestContext requestContext) {
        return new UserIsProjectOwnerValidator(action.getProjectId(),
                                               requestContext.getUserId(),
                                               projectDetailsManager);
    }

    @Nonnull
    @Override
    public RemoveProjectsFromTrashResult execute(@Nonnull RemoveProjectFromTrashAction action, @Nonnull ExecutionContext executionContext) {
        List<ProjectMovedFromTrashEvent> events = new ArrayList<>();
        ProjectId projectId = action.getProjectId();
        projectDetailsManager.setInTrash(projectId, false);
        events.add(new ProjectMovedFromTrashEvent(projectId));
        return new RemoveProjectsFromTrashResult(new EventList<>(EventTag.getFirst(), events, EventTag.getFirst()));
    }
}
