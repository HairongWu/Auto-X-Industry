package edu.stanford.bmir.protege.web.server.dispatch.impl;

import edu.stanford.bmir.protege.web.server.app.UserInSessionFactory;
import edu.stanford.bmir.protege.web.server.dispatch.*;
import edu.stanford.bmir.protege.web.server.project.ProjectManager;
import edu.stanford.bmir.protege.web.shared.app.UserInSession;
import edu.stanford.bmir.protege.web.shared.dispatch.Action;
import edu.stanford.bmir.protege.web.shared.dispatch.ActionExecutionException;
import edu.stanford.bmir.protege.web.shared.dispatch.Result;
import edu.stanford.bmir.protege.web.shared.permissions.PermissionDeniedException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 20/02/15
 */
@RunWith(MockitoJUnitRunner.class)
public class DispatchServiceExecutorImpl_TestCase<A extends Action<R>, R extends Result> {

    @Mock
    private ApplicationActionHandlerRegistry registry;

    @Mock
    private ActionHandler<A, R> actionHandler;

    @Mock
    private A action;

    @Mock
    private RequestContext requestContext;

    @Mock
    private ExecutionContext executionContext;

    @Mock
    private RequestValidator requestValidator;

    private DispatchServiceExecutorImpl executor;

    @Mock
    private ProjectManager projectManager;

    @Mock
    private UserInSessionFactory userInSessionFactory;

    @Mock
    private UserInSession userInSession;

    @Before
    public void setUp() throws Exception {
        executor = new DispatchServiceExecutorImpl(registry, projectManager, userInSessionFactory);
        when(registry.getActionHandler(action)).thenReturn(actionHandler);
        when(actionHandler.getRequestValidator(action, requestContext)).thenReturn(requestValidator);
        when(requestValidator.validateAction()).thenReturn(RequestValidationResult.getValid());
        when(userInSessionFactory.getUserInSession(any())).thenReturn(userInSession);
    }

    @Test(expected = ActionExecutionException.class)
    public void shouldThrowActionExecutionException() {
        ExecutionContext executionContext = this.executionContext;
        when(actionHandler.execute(action, executionContext)).thenThrow(new RuntimeException("Exception as part of test"));
        executor.execute(action, requestContext, executionContext);
    }

    @Test(expected = ActionHandlerNotFoundException.class)
    public void shouldThrowActionHandlerNotFoundException() {
        when(registry.getActionHandler(action)).thenThrow(new ActionHandlerNotFoundException(action));
        executor.execute(action, requestContext, executionContext);
    }

    @Test(expected = PermissionDeniedException.class)
    public void shouldThrowPermissionDeniedException() {
        when(requestValidator.validateAction()).thenReturn(RequestValidationResult.getInvalid("Denied"));
        executor.execute(action, requestContext, executionContext);
    }

    @Test
    public void shouldExecuteAction() {
        executor.execute(action, requestContext, executionContext);
        verify(actionHandler, times(1)).execute(action, executionContext);
    }
}
