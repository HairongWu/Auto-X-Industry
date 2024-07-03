package edu.stanford.bmir.protege.web.server.projectsettings;

import edu.stanford.bmir.protege.web.server.access.AccessManager;
import edu.stanford.bmir.protege.web.server.dispatch.ExecutionContext;
import edu.stanford.bmir.protege.web.server.project.ProjectDetailsManager;
import edu.stanford.bmir.protege.web.server.project.ProjectManager;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.shared.projectsettings.GetProjectSettingsAction;
import edu.stanford.bmir.protege.web.shared.projectsettings.GetProjectSettingsResult;
import edu.stanford.bmir.protege.web.shared.projectsettings.ProjectSettings;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.Mockito.when;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 25/11/14
 */
@RunWith(MockitoJUnitRunner.class)
public class GetProjectSettingsActionHandler_TestCase {


    private GetProjectSettingsActionHandler actionHandler;


    @Mock
    private ProjectId projectId;

    @Mock
    private ProjectSettings projectSettings;

    @Mock
    private ProjectDetailsManager mdm;

    @Mock
    private GetProjectSettingsAction action;

    @Mock
    private ExecutionContext executionContext;

    @Mock
    private ProjectManager projectManager;

    @Mock
    private AccessManager accessManager;

    @Before
    public void setUp() throws Exception {
        actionHandler = new GetProjectSettingsActionHandler(accessManager, projectId, mdm);
        when(mdm.getProjectSettings(projectId)).thenReturn(projectSettings);
    }

    @Test
    public void shouldReturnSettings() {
        GetProjectSettingsResult result = actionHandler.execute(action, executionContext);
        assertThat(result.getProjectSettings(), is(projectSettings));
    }
}
