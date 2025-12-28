package com.lelloman.simpleai

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for MainViewModel-related classes.
 *
 * Note: MainViewModel itself extends AndroidViewModel and requires Android framework
 * components (Application, ServiceConnection, BroadcastReceiver) which cannot be
 * easily mocked in pure JVM unit tests. Full ViewModel testing would require
 * instrumented tests or significant refactoring to abstract Android dependencies.
 *
 * These tests focus on the AppState sealed class and related data structures.
 */
class MainViewModelTest {

    @Test
    fun `AppState Starting is singleton`() {
        val state1 = AppState.Starting
        val state2 = AppState.Starting
        assertSame(state1, state2)
    }

    @Test
    fun `AppState Loading is singleton`() {
        val state1 = AppState.Loading
        val state2 = AppState.Loading
        assertSame(state1, state2)
    }

    @Test
    fun `AppState Ready is singleton`() {
        val state1 = AppState.Ready
        val state2 = AppState.Ready
        assertSame(state1, state2)
    }

    @Test
    fun `AppState Downloading holds progress value`() {
        val state = AppState.Downloading(0.5f)
        assertEquals(0.5f, state.progress)
    }

    @Test
    fun `AppState Downloading with different progress are not equal`() {
        val state1 = AppState.Downloading(0.3f)
        val state2 = AppState.Downloading(0.7f)
        assertNotEquals(state1, state2)
    }

    @Test
    fun `AppState Downloading with same progress are equal`() {
        val state1 = AppState.Downloading(0.5f)
        val state2 = AppState.Downloading(0.5f)
        assertEquals(state1, state2)
    }

    @Test
    fun `AppState Error holds message`() {
        val state = AppState.Error("Something went wrong")
        assertEquals("Something went wrong", state.message)
    }

    @Test
    fun `AppState Error with different messages are not equal`() {
        val state1 = AppState.Error("Error 1")
        val state2 = AppState.Error("Error 2")
        assertNotEquals(state1, state2)
    }

    @Test
    fun `AppState Error with same message are equal`() {
        val state1 = AppState.Error("Same error")
        val state2 = AppState.Error("Same error")
        assertEquals(state1, state2)
    }

    @Test
    fun `AppState sealed class covers all states`() {
        val states = listOf<AppState>(
            AppState.Starting,
            AppState.Downloading(0.5f),
            AppState.Loading,
            AppState.Ready,
            AppState.Error("test")
        )

        // Verify we can pattern match all states
        states.forEach { state ->
            when (state) {
                is AppState.Starting -> assertTrue(true)
                is AppState.Downloading -> assertTrue(state.progress >= 0)
                is AppState.Loading -> assertTrue(true)
                is AppState.Ready -> assertTrue(true)
                is AppState.Error -> assertTrue(state.message.isNotEmpty())
            }
        }
    }

    @Test
    fun `AppState Downloading progress can be 0`() {
        val state = AppState.Downloading(0f)
        assertEquals(0f, state.progress)
    }

    @Test
    fun `AppState Downloading progress can be 1`() {
        val state = AppState.Downloading(1f)
        assertEquals(1f, state.progress)
    }

    @Test
    fun `AppState types are distinguishable`() {
        val starting: AppState = AppState.Starting
        val downloading: AppState = AppState.Downloading(0.5f)
        val loading: AppState = AppState.Loading
        val ready: AppState = AppState.Ready
        val error: AppState = AppState.Error("test")

        assertTrue(starting is AppState.Starting)
        assertFalse(starting is AppState.Downloading)
        assertFalse(starting is AppState.Loading)
        assertFalse(starting is AppState.Ready)
        assertFalse(starting is AppState.Error)

        assertTrue(downloading is AppState.Downloading)
        assertTrue(loading is AppState.Loading)
        assertTrue(ready is AppState.Ready)
        assertTrue(error is AppState.Error)
    }

    @Test
    fun `AppState data classes have correct toString`() {
        val downloading = AppState.Downloading(0.75f)
        assertTrue(downloading.toString().contains("0.75"))

        val error = AppState.Error("Network failure")
        assertTrue(error.toString().contains("Network failure"))
    }

    @Test
    fun `AppState data classes support copy`() {
        val original = AppState.Downloading(0.5f)
        val copied = original.copy(progress = 0.8f)

        assertEquals(0.5f, original.progress)
        assertEquals(0.8f, copied.progress)
    }

    @Test
    fun `AppState Error message can be empty`() {
        val state = AppState.Error("")
        assertEquals("", state.message)
    }

    @Test
    fun `AppState can be used in when expression exhaustively`() {
        fun handleState(state: AppState): String = when (state) {
            is AppState.Starting -> "starting"
            is AppState.Downloading -> "downloading ${state.progress}"
            is AppState.Loading -> "loading"
            is AppState.Ready -> "ready"
            is AppState.Error -> "error: ${state.message}"
        }

        assertEquals("starting", handleState(AppState.Starting))
        assertEquals("downloading 0.5", handleState(AppState.Downloading(0.5f)))
        assertEquals("loading", handleState(AppState.Loading))
        assertEquals("ready", handleState(AppState.Ready))
        assertEquals("error: test", handleState(AppState.Error("test")))
    }
}
