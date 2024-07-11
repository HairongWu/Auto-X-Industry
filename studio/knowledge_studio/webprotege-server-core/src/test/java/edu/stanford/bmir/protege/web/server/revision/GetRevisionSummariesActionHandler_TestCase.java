package edu.stanford.bmir.protege.web.server.revision;

import edu.stanford.bmir.protege.web.server.access.AccessManager;
import edu.stanford.bmir.protege.web.server.dispatch.ExecutionContext;
import edu.stanford.bmir.protege.web.shared.revision.GetRevisionSummariesAction;
import edu.stanford.bmir.protege.web.shared.revision.GetRevisionSummariesResult;
import edu.stanford.bmir.protege.web.shared.revision.RevisionSummary;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.mockito.Mockito.when;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 21/02/15
 */
@RunWith(MockitoJUnitRunner.class)
public class GetRevisionSummariesActionHandler_TestCase {

    private GetRevisionSummariesActionHandler handler;

    @Mock
    private GetRevisionSummariesAction action;

    @Mock
    private ExecutionContext executionContext;

    @Mock
    private RevisionSummary summaryA, summaryB, summaryC;

    @Mock
    private RevisionManager revisionManager;

    @Mock
    private AccessManager accessManager;

    @Before
    public void setUp() throws Exception {
        List<RevisionSummary> revisionSummaries = new ArrayList<>();
        revisionSummaries.add(summaryA);
        revisionSummaries.add(summaryB);
        revisionSummaries.add(summaryC);
        handler = new GetRevisionSummariesActionHandler(accessManager, revisionManager);
        when(revisionManager.getRevisionSummaries()).thenReturn(revisionSummaries);
    }

    @Test
    public void shouldReturnRevisionSummaries() {
        GetRevisionSummariesResult result = handler.execute(action, executionContext);
        assertThat(result.getRevisionSummaries(), contains(summaryA, summaryB, summaryC));
    }
}
