package edu.stanford.bmir.protege.web.server.user;

import com.mongodb.MongoClient;
import edu.stanford.bmir.protege.web.server.project.RecentProjectRecord;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.shared.user.UserId;
import org.hamcrest.Matchers;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mongodb.morphia.Datastore;
import org.mongodb.morphia.Morphia;

import java.util.Optional;
import java.util.UUID;

import static edu.stanford.bmir.protege.web.server.persistence.MongoTestUtils.*;
import static java.util.Collections.singletonList;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 12 Mar 2017
 */
@SuppressWarnings("OptionalGetWithoutIsPresent")
public class UserActivityManager_IT {

    public static final long LAST_LOGIN = 33L;

    public static final long NEW_LAST_LOGIN = 88L;

    public static final long LAST_LOGOUT = 44L;

    public static final long NEW_LAST_LOGOUT = 99L;

    public static final long RECENT_PROJECT_TIMESTAMP = 77L;

    private Datastore datastore;

    private MongoClient mongoClient;

    private UserActivityManager repository;

    private UserId userId = UserId.getUserId("John Smith" );

    private ProjectId projectId = ProjectId.get(UUID.randomUUID().toString());

    private UserActivityRecord record = new UserActivityRecord(userId,
                                                           LAST_LOGIN,
                                                           LAST_LOGOUT,
                                                           singletonList(
                                                                   new RecentProjectRecord(projectId, 55L)));


    @Before
    public void setUp() throws Exception {
        mongoClient = createMongoClient();
        Morphia morphia = createMorphia();
        datastore = morphia.createDatastore(mongoClient, getTestDbName());
        repository = new UserActivityManager(datastore);
    }

    @After
    public void tearDown() throws Exception {
        mongoClient.dropDatabase(getTestDbName());
        mongoClient.close();
    }

    @Test
    public void shouldSaveUserActivityRecord() {
        repository.save(record);
        assertThat(datastore.getCount(UserActivityRecord.class), is(1L));
    }

    @Test
    public void shouldFindUserActivityRecord() {
        repository.save(record);
        assertThat(repository.getUserActivityRecord(userId), is(Optional.of(record)));
    }

    @Test
    public void shouldSetLastLogin() {
        long lastLogin = NEW_LAST_LOGIN;
        repository.setLastLogin(userId, lastLogin);
        Optional<UserActivityRecord> record = repository.getUserActivityRecord(userId);
        assertThat(record.get().getLastLogin(), is(lastLogin));
    }

    @Test
    public void shouldSetLastLogout() {
        long lastLogOut = NEW_LAST_LOGOUT;
        repository.setLastLogout(userId, lastLogOut);
        Optional<UserActivityRecord> record = repository.getUserActivityRecord(userId);
        assertThat(record.get().getLastLogout(), is(lastLogOut));
    }

    @Test
    public void shouldAddRecentProject() {
        long timestamp = RECENT_PROJECT_TIMESTAMP;
        repository.addRecentProject(userId, projectId, timestamp);
        Optional<UserActivityRecord> record = repository.getUserActivityRecord(userId);
        assertThat(record.get().getRecentProjects(), Matchers.contains(new RecentProjectRecord(projectId,
                                                                                               timestamp)));
    }


}
